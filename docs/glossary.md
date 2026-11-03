# Glossary & Mental Models

New to **[DiffusionGemma](https://gemma.ai/models/diffusiongemma/) (`dgem`)**? Because `dgem` sits at the intersection of **Classical Search/Tabular ML**, **Formal Language Automata**, and **Discrete Diffusion Generative Models**, engineers arriving from different specialties often bring different terminology for overlapping ideas.

Use this page as a **Decoder Ring** to translate between disciplines.

---

## 1. Quick-Start Mental Model Matrix (By Reader Background)

| If You Come From... | Think of `dgem` (`steps=1, think=0`) As... | Think of `.json.tmpl` Templates As... | Think of the Entropy Cascade As... |
| :--- | :--- | :--- | :--- |
| **Software / Platform Engineering** | A sub-second type-safe RPC call that returns guaranteed JSON booleans, enums, and scores without hallucinated syntax. | **Policy-as-Code** (like OpenPolicyAgent `.rego` or JSON Schema, but for semantic natural language & code rules). | An automatic circuit breaker that keeps easy traffic on the fast path and escalates only hard edge cases. |
| **Search, Ranking & Tabular ML (`GTR`, `DeBERTa`, `TabPFN`)** | A **zero-shot multi-head Cross-Encoder** where all target heads (`slot_1 <-> slot_2`) mutually attend in 1 forward pass with zero training rows ($N=0$). | A declarative classification head compiler—changing a label set is an edit, not relabeling + retraining. | A scale-invariant **Selective Prediction / Abstention Gate** ($\tilde{H}_m = H_m / \ln|\mathcal{V}_m|$) paired with Bayesian prior forwarding. |
| **LLM / GenAI Infrastructure (`vLLM`, `CoT`)** | Replacing an $O(L)$ sequential token generation loop with an **$O(1)$ parallel block-diffusion logit readout** over masked slot positions. | Constrained single-token decoding (`[A–Z]`, `yes/no`, `1..5`) evaluated simultaneously across all slots. | **Adaptive Test-Time Compute (`think=0 -> think>0`)**: only spending reasoning scratchpad tokens when Pass-1 epistemic entropy spikes. |

---

## 2. Decision Model & Canvas Primitives

### Discrete Diffusion Slot Readout
* **In Plain English**: Reading the exact probability of every valid answer (`yes`/`no`, `A`–`Z`, `1`–`5`) directly from masked blanks in a single forward pass instead of generating words one by one. Each option is given a letter and each blank holds one letter, which is why a `choice` question has at most 26 options (wider lists are split into rounds, or use the opt-in 52-label sets). Hesitation is computed from these answer probabilities.
* **Under the Hood**: Given a causal prompt prefix and a seeded canvas containing masked slot positions $\langle s_1, \dots, s_M \rangle$, the engine runs 1 denoising step (`steps=1, think=0`), slices the raw vocabulary logits $z_{m,k}$ restricted to the valid single-token options $k \in \mathcal{V}_m$, and normalizes via softmax.
* **Where You See It in `dgem`**: [`pkg/client/client.go`](../pkg/client/client.go) (`ParseStructuredContentWithLogprobs`) and `structured_server.py`.

### Bidirectional Canvas Attention
* **In Plain English**: Every input word directly inspects every policy rule and every decision blank—and the decision blanks inspect *each other* simultaneously (`slot_1 <-> slot_2`).
* **Under the Hood**: Standard LLM decoders apply a lower-triangular causal mask ($t$ can only see $<t$). [DiffusionGemma](https://gemma.ai/models/diffusiongemma/) uses a hybrid attention mask (`TRITON_ATTN` in vLLM): causal over the prompt prefix (for KV-cache reuse) and **all-to-all bidirectional** over the answer canvas (served with 128 rows, 32 on Cloud Run L4; each request uses the smallest multiple of 16 rows that holds its answer template).
* **Where You See It in `dgem`**: Enables joint 3-slot triage (`urgent` + `team` + `sentiment`) in one pass.

### Policy-as-Template (`.json.tmpl`)
* **In Plain English**: A declarative JSON file where you define decision questions (`boolean`, `choice`, `score`) and natural-language rubrics that execute immediately with zero model training.
* **Under the Hood**: Go `text/template` files compiled by `dgem decide -t` into structured slot schemas and single-token option maps (`[A–Z]`).
* **Where You See It in `dgem`**: [`templates/`](../templates) (`templates/calibration/*.json.tmpl`, `templates/secops_conditional_dag.json.tmpl`).

### Prompt Layout (`document_first` / `schema_first`)
* **In Plain English**: Whether the model reads your input before or after it sees the questions. Like handing someone the letter first and then the questionnaire, instead of the questionnaire first.
* **Under the Hood**: `document_first` (default from serving v0.2.0) puts the input in `<user_text>` and the questions in `<instructions>`, both in the user turn; `schema_first` sends the questions as the system message. Set per template, request (`"layout"`), CLI (`--layout`), Studio or deployment (`DEFAULT_LAYOUT`).
* **Where You See It in `dgem`**: [Prompt layout guide](policies/prompt-layout.md), `diagnostics.layout` in responses.

### Conditional Policy DAG (`depends_on` & `ask_if`)
* **In Plain English**: A multi-stage decision flowchart where follow-up questions are only evaluated if an upstream gate question resolves to `true` (or a specific option).
* **Under the Hood**: `schedule()` in the serving image's `structured_server.py` sorts questions topologically into stages; each stage after the first is one more sequential read. If Stage 1 (`is_prompt_injection`) evaluates to `false`, downstream forensic slots are pruned after **1 pass**, saving the second pass on benign traffic.
* **Where You See It in `dgem`**: `templates/secops_conditional_dag.json.tmpl`; [templates guide](policies/templates.md).

---

## 3. Uncertainty & Cascade Telemetry

### Relative Tie-Detection vs. Target-Domain Probability Calibration
* **In Plain English**: `dgem`'s single-pass logprob scores tell you **whether the model is torn between your template choices** (relative routing ambiguity), *not* the real-world base rate of how often a class appears in your database.
* **Why This Matters**: True statistical calibration ($P(\text{Gold}=A \mid \hat{p}=0.80) = 0.80$) depends on the target environment's class prior $P_{\text{target}}(Y)$ and always requires post-hoc target data (Platt scaling, temperature scaling, or conformal prediction). What `dgem` provides zero-shot in 1 forward pass is a **tie-detector over the user-supplied option letters (`A..Z`)**—eliminating the $10\times\text{–}50\times$ token-cost multiplier of multi-sample autoregressive confidence rollouts. Order bias can also make a tie look decisive; see [Confidence beyond Shannon](confidence/overview.md) for how `dgem` monitors it and gates on hesitation.

### Distributional Discrete Regression (`score` Slots)
* **In Plain English**: Turning continuous regression (like a `1..5` severity score) into a probability histogram over discrete levels so that classification confidence and regression variance come out of the exact same softmax formula.
* **Under the Hood**: Instead of a point-estimate MSE head or quantile pinball loss, `dgem` evaluates the restricted-softmax probabilities $p_{m,1}, \dots, p_{m,L}$ over the discrete numeric bins $v_1, \dots, v_L$ (`pkg/client/client.go`), yielding the continuous expected value $\mathbb{E}[v] = \sum_k v_k p_{m,k}$, ordinal variance $\text{Var}(v) = \sum_k p_{m,k}(v_k - \mathbb{E}[v])^2$, and normalized entropy $\tilde{H}_m = H(p) / \ln L$ in 1 pass.
* **Where You See It in `dgem`**: Every `score` primitive in `.json.tmpl` templates (`sentiment`, `risk_score`, `severity`).

### Cardinality-Normalized Entropy
* **In Plain English**: A universal **`0.0` to `1.0` uncertainty meter** that adjusts for how many answer choices a question has (`2` options vs. `26` options).
* **Under the Hood**: Raw Shannon entropy $H_m = -\sum_{k \in \mathcal{V}_m} p_{m,k} \ln p_{m,k}$ has a theoretical maximum of $\ln|\mathcal{V}_m|$ (`0.693 nats` for binary vs. `3.258 nats` for 26-way `choice`). Dividing by $\ln|\mathcal{V}_m|$ yields the dimensionless normalized entropy:
  $$\tilde{H}_m = \frac{H_m}{\ln|\mathcal{V}_m|} \in [0, 1]$$
* **Where You See It in `dgem`**: `--normalize-entropy --cascade-threshold 0.16` in `dgem bench-calibration` (`cmd/bench_calibration.go`).
* **In Decision Studio**: shown as **Hesitation** (0–100%) on every answer: under 16% is *Clear*, 16–50% *Somewhat unsure*, above 50% *Very unsure*. Hover to see the raw entropy in nats. The 16% line matches the normalized cascade gate `--normalize-entropy --cascade-threshold 0.16`.

### Multi-Slot Scale Inversion
* **In Plain English**: The bug that happens when you apply a single raw entropy cutoff (like `0.35 nats`) to questions with different numbers of choices—causing confident 26-choice questions to falsely escalate while uncertain 3-choice questions slip through!
* **Under the Hood** *(illustrative numbers)*: On a 26-option intent question, tiny residual probabilities across 25 classes can yield $H \approx 0.52\text{ nats}$ even when the model is confident and right ($\tilde{H} \approx 0.16$). Meanwhile, on a 3-option NLI question, a real rise in uncertainty to $H \approx 0.25\text{ nats}$ is below `0.35 nats` in raw units, but **above `0.16` once normalized by $\ln(3)$** ($\tilde{H} \approx 0.23$).
* **Where You See It in `dgem`**: Avoided with `--normalize-entropy` (`dgem bench-calibration`) and the Hesitation % scale in the Studio.

### Pass-1 Slot Prior Forwarding
* **In Plain English**: Handing Stage 2 not just the original question, but also Stage 1's exact probability breakdown (`{entailment: 94.2%, neutral: 4.9%, contradiction: 0.9%}`) as a diagnostic clue to double-check.
* **Under the Hood**: `formatTier1PriorBlock` (`cmd/bench_calibration.go`) injects `[TIER-1 DISCRETE DIFFUSION PRIOR TELEMETRY]` sorted by restricted-softmax probability descending, acting as a **cognitive counter-anchor** that forces Stage 2 to verify why `dgemma`'s entropy spiked before committing to a label.
* **Where You See It in `dgem`**: Stage 2 cascades (`cascade_mode: entropy`) and `dgem bench-calibration --cascade-from`.

---

## 4. Hesitation-Gating & Trust Techniques

These terms come from [Confidence beyond Shannon: hesitation-gated decisions](confidence/overview.md), which explains each in more detail.

### Hesitation-Gating
* **In Plain English**: Answer directly when the model is clearly sure; hand the decision to a larger model or a person when it hesitates. Like a triage nurse who treats clear cases and refers the rest.
* **Under the Hood**: Escalate when hesitation (normalized entropy $H/\ln K$) is at or above a threshold, forwarding `dgem`'s probabilities as a hint. Fit the threshold on labelled examples from your own domain.
* **Where You See It in `dgem`**: `cascade_mode` / `cascade_threshold` (in nats, or hesitation with `cascade_threshold_mode: "normalized"`) / `cascade_model` on the gateway, MCP and Studio; Hesitation % on every Studio answer.

### Answer Template Effects (Letter Collision & Slot Names)
* **In Plain English**: The model reads the whole answer form, labels and question names included. If two questions use the same letters for different options, it tends to copy the letter across; if a question's name hints the answer should differ, it does.
* **Under the Hood**: In our tests, a reversed second slot labelled with the same letters, or a second slot named `…__mirror_rev`, measurably lowered accuracy even with identical options; digit labels and neutral ids avoided it.
* **Where You See It in `dgem`**: Template rule: neutral, descriptive question ids; no shared letters with different meanings ([templates](policies/templates.md)).

### Ballot-Order (Primacy) Bias / Null Prior $p_0$
* **In Plain English**: Like undecided voters who tick the first name on a ballot, the model leans toward whichever option is listed first (`A`). Given a blank question with meaningless options, [`DiffusionGemma`](https://gemma.ai/models/diffusiongemma/) still picks `A` far more often than chance, especially with only 2 or 3 options.
* **How dgem handles it**: Measured on every release (the regression matrix shuffles option order and reports extra flips) rather than corrected in the pass; see the research notes below.
* **Under the Hood**: $p_0(k)$ is the model's slot distribution on a content-free input. It estimates the position term $b_{\text{pos}}(k)$ in $z = s(\text{option}) + b_{\text{pos}}(k) + \epsilon$.

### Temperature Scaling ("Humility Dial") — calibrated per policy
* **In Plain English**: One dial that makes over-confident scores more modest (or under-confident ones bolder) without changing which answer wins. This is the "beyond Shannon" step: each policy gets its own dial, fitted on held-out labelled examples; there is no global setting.
* **Under the Hood**: $p_k(T) \propto p_k^{1/T}$. $T > 1$ softens. $T^*$ is **fitted on labeled examples** (Guo et al., 2017), so the improvement is only trustworthy when measured on data not used for fitting. The best held-out temperature differs noticeably by domain, so fit one per policy.

### Expected Calibration Error (`ECE`)
* **In Plain English**: "When the model says 80%, is it right about 80% of the time?" ECE is the average gap between stated confidence and actual accuracy, so `0` is perfect.
* **Under the Hood**: Predictions are grouped into 10 confidence bins; ECE $= \sum_b \frac{n_b}{N}\,|\text{acc}_b - \text{conf}_b|$. It's noisy on small datasets, especially when most predictions fall in the top bin.

### Brier Score
* **In Plain English**: A penalty for being confidently wrong *and* for being needlessly unsure when right. Lower is better. A perfect, fully confident forecaster scores `0`.
* **Under the Hood**: $\text{Brier} = \frac1N \sum_i \sum_k (p_{i,k} - y_{i,k})^2$, where $y$ is the one-hot gold label.

### Research Notes: Order-Bias Corrections

The techniques below were tested to correct order bias inside the pass and are kept as research tools. None improved decisions reliably in our tests.

#### Null-Prior De-Biasing ("Tare the Scale")
* **In Plain English**: Weigh the empty bowl first, then subtract it. `dgem` divides out the model's built-in preference for each slot before reporting confidence. It needs no labeled data.
* **Under the Hood**: $\tilde{p}_k \propto p_k / p_0(k)^{\alpha}$, with $\alpha \in [0,1]$ controlling correction strength (`--prior-alpha`, default `0.5`). Related prior work: *contextual calibration* (Zhao et al., 2021).
* **Caveat**: It removes the *average* slot habit, not input-specific order effects. In our tests it could improve Brier score while increasing flips under other orderings, and it helped some suites while making calibration worse on others, so treat it as suite-dependent.

#### Dual-Mirror Canvas
* **In Plain English**: Print the ballot twice on the same page, once in reverse order, and check that both votes agree. Because a diffusion model fills every blank at once, the second copy costs no extra forward pass.
* **Under the Hood**: For each `choice` slot, `dgem` adds `<id>__rev` with options $[o_K \dots o_1]$, reads both in one pass, maps them back to option names, and merges them (currently 70% forward / 30% reversed with a forward-priority rule).
* **Caveats**: The two slots can see each other on the canvas, so they are not independent readings. Reversal is only one reordering: an item can flip under a cyclic shift but pass the mirror check. In our tests the extra slot lowered forward accuracy, and the original slot id `__mirror_rev` degraded readings further (renamed `__rev`). Treat it as a research diagnostic. The damage comes from letter collision (both slots labelled A, B, C… with different meanings); a digit-labelled mirror (`--mirror-mode reversed-digits`) avoids it.

#### Mirror TVD (Total Variation Distance)
* **In Plain English**: How far apart the forward and reversed readings are, from `0` (identical) to `1` (completely different). Near 0 means order didn't matter for this input. A large value means the confidence depends on the layout.
* **Under the Hood**: $\text{TVD} = \tfrac12 \sum_k |p^{\text{fwd}}_k - p^{\text{rev}}_k|$. Clear-cut items typically score near `0`; an order-sensitive item can score well above that while its single reading looks near-certain.
* **Status**: Computed in `bench-permutation`. Not yet returned by `dgem decide`, `dgem serve`, MCP, or the Studio.

#### Cyclic JSD (Permutation Mutual Information)
* **In Plain English**: The expensive, thorough version of the mirror: ask the question once per rotation of the option list and measure how much the answers disagree.
* **Under the Hood**: The Jensen–Shannon divergence across $K$ cyclic orderings estimates $I(Y; \Pi \mid X)$, the information the option order carries about the answer. It costs $K$ forward passes.


---

## 5. Comparative ML Architectures

### Dual Encoder (`GTR` / `Sentence-T5`)
* **In Plain English**: A bi-encoder architecture that compresses the input text into one vector $u$ and the label description into another vector $v_k$ *independently*, then compares the two vectors at the very end.
* **Under the Hood**: Because $u = E_x(x) \in \mathbb{R}^d$ is computed **before** the model sees the policy rules or hypothesis $c_k$, token-to-token alignment (like checking whether a specific SQL argument matches an allowlist or comparing `50–75%` against `100%`) is lost during vector pooling (**Late Interaction Bottleneck**).
* **Where You See It in `dgem`**: Contrasted with `dgem`'s early all-to-all cross-attention in [Discrete Diffusion vs. Autoregression (§5)](confidence/architecture.md#5-architectural-faq-can-dual-encoders-gtr--tabpfn-replace-a-decision-model-or-do-you-need-test-time-compute).

### `TabPFN` & Tabular Foundation Models
* **In Plain English**: A foundation model pre-trained on millions of synthetic spreadsheets that predicts a missing target column by attending across labeled example rows (`in-context learning` for tables).
* **Under the Hood**: `TabPFN` approximates Bayesian posterior inference $P(y_{\text{test}} \mid X_{\text{test}}, X_{\text{train}}, y_{\text{train}})$ in a single forward pass. However, it requires **labeled support rows ($N_{\text{support}} > 0$)** in its context window and operates on pre-extracted tabular columns—meaning pairing `GTR + TabPFN` still suffers from `GTR`'s pooling bottleneck and cannot compile zero-shot (`N=0`) natural-language `.json.tmpl` policies.

### Fixed-Depth Circuits ($\mathsf{TC}^0$) vs. Test-Time Compute
* **In Plain English**: Why a single forward pass (`think=0`) can verify direct relational facts, whereas multi-step mental arithmetic (`2015 + 4 = 2019 > 2018`) requires generating scratchpad tokens (`think > 0`).
* **Under the Hood**: A transformer with fixed layer depth $L$ and no scratchpad generation (`think=0`) is bounded by the circuit complexity class $\mathsf{TC}^0$. When a contradiction depends on an intermediate state not present in the input text (`anli-02`'s latent year `2019`), test-time compute (`--cascade-self-think 256` or Tier-2 reasoning) allocates working-memory tokens to materialize the intermediate state.

---

## 6. Spatial Grounding & Vision-Language Terminology

### `DETR` Object Queries (Detection Transformer)
* **In Plain English**: Instead of scanning an image with thousands of sliding-window guesses and filtering duplicates afterward (`Non-Maximum Suppression`), `DETR` creates a fixed number of parallel "empty parking spots" (**Object Queries**—e.g., `obj1` and `obj2`). Because all query slots attend to the image and to **each other simultaneously**, `obj2` sees that `obj1` already claimed the left object and automatically claims the right object in a single pass.
* **Under the Hood**: In `dgem`, `templates/multimodal/bbox_multi_object_detr.json.tmpl` places `obj1_[ymin,xmin,ymax,xmax]` and `obj2_[ymin,xmin,ymax,xmax]` on the same bidirectional `[MASK]` canvas (`reads=1`), allowing the query slots to co-adapt without autoregressive left-to-right drift.
* **Where You See It in `dgem`**: `dgem bench-bbox` (`bbox-t3-01-detr-dual-buttons`, `bbox-t3-02-detr-stacked-banner-cta`).

### Softmax Expectation (probability-weighted coordinate)
* **In Plain English**: Instead of taking only the single most likely coordinate label (`argmax`), take the **probability-weighted average** over all 21 labels (`00, 05, ..., 100`). If the model is split 50/50 between `30` and `35`, the expectation says `32.5`.
* **Under the Hood**:
  $$\hat{c}_m = \sum_{k=0}^{20} (5k) \cdot P(\text{slot}_m = \text{bin}_k)$$
* **In our tests**: The expectation box and the argmax box were about equally good. It slightly hurt boxes whose edges sit exactly on a label. It does not deliver sub-bin precision: most of the error comes from the model looking in slightly the wrong place, not from rounding.
* **Where You See It in `dgem`**: `cmd/bench_bbox.go` (`evaluateCoordinateDistribution`).

### Per-Edge Entropy ($\tilde{H}_{\text{edge}}$)
* **In Plain English**: Each side of the box (`ymin`, `xmin`, `ymax`, `xmax`) gets its own hesitation score, so you can see *which* side the model is unsure about.
* **Under the Hood**: $\tilde{H}_m = H_m / \ln K$ for each edge $m$, where $K$ is the number of labels (21).
* **In our tests**: High edge hesitation picked out edges that were more than one step wrong. It did **not** rise when an edge was hidden behind another element.
* **Where You See It in `dgem`**: `dgem bench-bbox` (edge table and "Entropy → edge error" line).

