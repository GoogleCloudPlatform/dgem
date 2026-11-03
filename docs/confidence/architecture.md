# Discrete Diffusion vs. Autoregression

## 1. The Bottleneck of Autoregressive LLMs

Standard Large Language Models (LLMs) operate under a sequential autoregressive factorization:

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

Each forward pass generates exactly **one token**. Even if the model only needs to output a single boolean decision or three fields of a JSON object:
1. It must sequentially predict structural characters (`{`, `\n`, `"`, `k`, `e`, `y`, `"`, `:`, ` `).
2. It suffers from high memory bandwidth pressure: loading billions of parameters from memory to compute a single token's logits.
3. It takes dozens or hundreds of forward passes, one per output token.

---

## 2. Discrete Block Diffusion & Multi-Canvas Sampling

**[DiffusionGemma](https://gemma.ai/models/diffusiongemma/) (`dgemma`)** breaks this sequential bottleneck by using **discrete block diffusion**:

* **Block-Autoregressive Canvas**: The decoder works on a short canvas (served with **128 rows**, 32 on Cloud Run L4; each request uses the smallest multiple of 16 that holds its answer template) with **bidirectional self-attention**.
* **Iterative Denoising**: The answer positions begin as seeded noise tokens and are denoised in parallel across a single pass (`steps: 1`, `think: 0`) or a small number of steps.
* **Joint Multi-Slot Conditioning (`slot_1 <-> slot_2`)**: Unlike independent classification heads, all masked decision slots attend to the prompt and to *each other* simultaneously in $O(1)$ forward passes.

---

## 3. How Discrete Diffusion Slot Readout Works

In a structured decision query (`steps: 1, think: 0`), no conversational prose is generated:

1. **Canvas Seeding**: The known policy template (e.g. `urgent: @\nteam: @\nsentiment: @`) is pre-seeded into the canvas, where `@` marks an answer position. The serving image fills each one with a random token (seeded per sample) and asks vLLM for a read-only step that returns the logprob of every allowed label at that position.
2. **Single-Pass Readout**: A single forward pass executes across the causal prompt prefix and bidirectional canvas.
3. **Restricted-Softmax Logit Readout**: Rather than decoding free-form text, the engine extracts the raw logits $z_{m,k}$ restricted to the valid single-token candidate vocabulary $\mathcal{V}_m$ (`{"yes","no"}` for `boolean`, `[A–Z]` for `choice`, `1..9` for `score`; each label must be one token at the same position, hence at most 26 options) and normalizes via softmax:

$$p_{m,k} = \frac{\exp(z_{m,k})}{\sum_{j \in \mathcal{V}_m} \exp(z_{m,j})}$$

```text
Seeded Canvas:
[<|channel>thought\n<channel|>urgent: @ \nteam: @ \nsentiment: @ ]
                                      ▲          ▲           ▲
                                   Slot 1     Slot 2      Slot 3
                              [p(yes), p(no)] [p(A)..p(Z)] [p(1)..p(5)]
                                |V_1| = 2      |V_2| = 26   |V_3| = 5
```

---

## 4. Uncertainty & Cardinality-Normalized Entropy

Single-pass restricted-softmax readout gives a probability distribution and an uncertainty signal at every decision slot $m$. These raw probabilities rank errors well but are somewhat over-confident, by an amount that depends on the domain; fit a temperature per policy on held-out data if you need calibrated probabilities (see [calibrate your policy](calibrate-your-policy.md#5-temperature-scaling-optional)):

* **Raw Shannon Entropy ($H_m$)**:
  $$H_m = -\sum_{k \in \mathcal{V}_m} p_{m,k} \ln p_{m,k} \in [0, \ln|\mathcal{V}_m|]$$
* **Cardinality-Normalized Epistemic Entropy ($\tilde{H}_m$)**: Because maximum entropy scales logarithmically with option count ($\ln 2 = 0.693\text{ nats}$ for binary `boolean` vs. $\ln 26 = 3.258\text{ nats}$ for 26-way `choice`), `dgem` normalizes each slot's entropy by its theoretical ceiling $\ln|\mathcal{V}_m|$:
  $$\tilde{H}_m = \frac{H_m}{\ln|\mathcal{V}_m|} \in [0, 1]$$

<details class="term-aside">
<summary>💡 <strong>Concept Aside: Why does Raw Entropy ($H_m$) cause "Multi-Slot Scale Inversion" without $\ln|\mathcal{V}_m|$ normalization?</strong> <em>(click to expand)</em></summary>

*Illustrative numbers.*

* **In Plain English**: A 26-option banking classifier naturally leaks tiny probability crumbs across 25 runner-up classes even when it is confident and right, inflating its raw entropy (say `0.52 nats`). Meanwhile, a 3-option NLI slot (`entailment` / `neutral` / `contradiction`) has a small maximum ceiling (`1.099 nats`), so a real spike in uncertainty (say `0.25 nats`) looks smaller in raw nats than the 26-way slot.
* **How $\tilde{H}_m = H_m / \ln|\mathcal{V}_m|$ Fixes It**: Dividing by $\ln|\mathcal{V}_m|$ puts every slot onto a universal $[0, 1]$ uncertainty scale (`--normalize-entropy --cascade-threshold 0.16`):
  * **26-option intent slot**: $0.52 / \ln(26) \approx \mathbf{0.16}$, at the threshold: a confident answer.
  * **3-option NLI slot**: $0.25 / \ln(3) \approx \mathbf{0.23 \ge 0.16} \implies$ escalates to Stage 2.
* **See also**: the [Glossary entry on Multi-Slot Scale Inversion](../glossary.md#multi-slot-scale-inversion).

</details>

* **Adaptive Sampling & Prior-Guided Escalation**:
  - **Low Normalized Entropy ($\tilde{H}_m < 0.160$)**: The decision is decisive. `dgem` early-exits immediately after **1 forward pass**.
  - **High Normalized Entropy ($\tilde{H}_m \ge 0.160$)**: `dgem` forwards the **Pass-1 Slot Prior Distribution (`[TIER-1 DISCRETE DIFFUSION PRIOR TELEMETRY]`)** to Stage 2—either an **Intra-Model Self-Cascade** (`--cascade-self-think 256` on the same `dgemma` GPU) or a **Cross-Model Cascade** (`gemini-3.8-flash`).

### How `dgem` Computes Confidence & Entropy from Template Logprobs (Step-by-Step)

In standard autoregressive LLM pipelines, estimating whether the model is confident on a custom task—without generating long Chain-of-Thought rationales or running $N=10\text{–}20$ Monte Carlo rollouts (self-consistency)—multiplies token costs by orders of magnitude.

`dgem` avoids that token-cost explosion by turning your `.json.tmpl` policy into a **single-token `logprob` tie-detector** across 4 concrete steps:

1. **Step 1 — Map User-Supplied Classes to Single Letters (`A`, `B`, `C`...) in the Prompt**:
   When you define custom classes in a template (even for a domain not in the model's training distribution), `dgem` formats them into a single-letter legend in the prompt prefix and places **one masked blank (`@`)** per question on the diffusion canvas:
   ```text
   Prompt Prefix:
     Slot 'intent' options:
       A = billing_dispute
       B = account_compromise
       C = feature_request

   Seeded Diffusion Canvas (1 token per slot):
     intent: @
   ```
2. **Step 2 — Run 1 Forward Pass (`think=0`) & Read `logprobs` of *Only* Those Letters**:
   Instead of generating free-form text, `dgem` executes **1 forward pass** and inspects the raw token logits at that exact `@` blank. It discards the other ~255,997 words in the vocabulary and runs a softmax strictly over the valid letters (`A`, `B`, `C`) you supplied:
   $$p_A, p_B, p_C = \text{softmax}(z_A, z_B, z_C)$$
   *(For numeric `score` slots like `1..5`, it does the exact same thing over the digit tokens `'1'..'5'`, computing the weighted average $\mathbb{E}[v] = \sum_{k=1}^5 k \cdot p_k$ and spread from those 5 probabilities).*
3. **Step 3 — Measure Whether the Top Letters Are in a Close Race ($H_m$)**:
   - If the restricted probabilities are **`{A: 97.5%, B: 1.5%, C: 1.0%}`**, letter `A` dominates ($H_m = 0.13\text{ nats}$). We take `A` immediately and pay **zero** generation tokens.
   - If the restricted probabilities are **`{A: 54.0%, B: 42.0%, C: 4.0%}`**, the model's attention is **torn** between `A` and `B` ($H_m = 0.82\text{ nats}$).
4. **Step 4 — Divide by $\ln(\text{Number of Choices})$ to Scale the "Tie Meter" from `0.0` to `1.0` ($\tilde{H}_m$)**:
   Why can't we use the same raw entropy cutoff ($H_m$) for a 2-choice `yes/no` question and a 26-choice `A..Z` question? Because a flat dead tie between 2 choices has a maximum entropy of $\ln(2) = 0.693$, while a flat tie between 26 choices has a maximum entropy of $\ln(26) = 3.258$. Dividing by $\ln(\text{number of choices})$:
   $$\tilde{H}_m = \frac{H_m}{\ln(\text{number of choices})} \in [0, 1]$$
   scales our "tie meter" onto `0.0` (one letter dominates) to `1.0` (dead tie) regardless of how many classes you put in your template. When $\tilde{H}_m \ge 0.16$, `dgem` escalates the query to a reasoning pass (`think > 0` or Tier-2 LLM) **and passes along the Pass-1 letter breakdown (`{A: 54%, B: 42%}`)** so the reasoning model knows which two candidates to disambiguate.

:::caution[Important Statistical Distinction: Relative Tie-Detection vs. Target-Domain Base-Rate Calibration]
In statistics and classical ML, **true probability calibration** means that when a classifier outputs `0.80` for class `A`, class `A` empirically occurs `80%` of the time in your production environment.

**No zero-shot model can provide true out-of-the-box base-rate calibration on an unseen task without target-domain data**, because the model does not know your production class priors $P_{\text{target}}(Y)$ (e.g., whether a rare clinical or fraud event happens in `0.1%` or `30%` of cases). If your application requires calibrated frequentist probabilities tied to production base rates, you still need a post-hoc calibration layer fit on labeled target examples (such as Platt/temperature scaling, isotonic regression, or conformal prediction).

What `dgem`'s normalized entropy $\tilde{H}_m$ provides zero-shot is **Relative Routing Ambiguity (Tie-Detection)**—answering *"Given the options in the template, does one choice clearly win in 1 forward pass, or are the top choices competing?"*—allowing you to early-exit clear traffic in a single pass and reserve expensive autoregressive reasoning tokens for the inputs where the choices are in contention.
:::

---

## 5. Architectural FAQ: Can Dual-Encoders (`GTR`) + `TabPFN` Replace a Decision Model, or Do You Need Test-Time Compute?

Engineers from search, retrieval, and tabular ML backgrounds frequently ask a foundational design question:

> *"Could the goal of a fast, reasoning-capable classifier be achieved without a generative model—specifically by pairing a **GTR-style Dual Encoder** (`Sentence-T5`) with a **TabPFN / TabFM** zero-shot tabular classification foundation model? Or do you strictly need a decoder and test-time compute (`think > 0`) to pull off reasoning?"*

:::tip[TL;DR: The 30-Second Architectural Answer]
1. **Why `GTR Dual-Encoder + TabPFN` Hits an Early Information Wall**: A Dual Encoder compresses your entire input document into a single fixed vector $u \in \mathbb{R}^d$ **before** reading your policy rules or hypothesis. That pooling step permanently destroys token-to-token relational alignment (such as negation scope, numerical bounds like `50–75% < 100%`, or SQL parameter tampering in `AgentDrift`). Feeding those pooled embeddings into `TabPFN` cannot recover fine-grained relational bindings already lost in $u$—and `TabPFN` further requires **labeled support rows ($N_{\text{support}} > 0$)** rather than zero-shot instructions.
2. **When You Do *NOT* Need a Decoder (`think=0` in $O(1)$ — most tasks)**: Full **token-level cross-attention** across a 26B-A4B model (`dgem` with `steps=1, think=0`) handles 1-hop relational policy grounding (tool-call audits, prompt-injection detection, fact grounding, passage relevance) in **a single forward pass** without generating a single scratchpad token.
3. **When You *DO* Need a Decoder + Test-Time Compute (`think > 0` — `anli-01..03`)**: When a decision hinges on synthesizing an **unwritten intermediate variable** (such as computing `2015 + 4 = 2019` and checking `2019 > 2018` in `anli-02`), constant-depth circuit bounds ($\mathsf{TC}^0$) prevent *any* single-pass model (`GTR`, `TabPFN`, or `dgemma [think=0]`) from reliably chaining the arithmetic. Because `dgemma` is a unified architecture, its **normalized entropy gate ($\tilde{H}_m \ge 0.160$)** detects those exact multi-hop traps and triggers `think > 0` (conditioned on Pass-1 priors) only on the queries that need a scratchpad.
:::

### The Operative Decoder Ring (5 Core Concepts in Plain English)

For readers arriving from different specialties (Platform Engineering, Search/Retrieval, or LLM Infrastructure), here is how the five architectural terms map to plain English:

| Term | 10-Word Plain-English Mental Model | Canonical Example |
| :--- | :--- | :--- |
| **[Dual Encoder (`GTR` / `T5`)](../glossary.md#dual-encoder-gtr--sentence-t5)** | Compresses input and label into two separate vectors, then compares. | Semantic search & topical intent (`Banking77` similarity). |
| **[Tabular FM (`TabPFN` / `TabFM`)](../glossary.md#tabpfn--tabular-foundation-models)** | Predicts a spreadsheet column by attending to labeled example rows. | Few-shot classification over numerical/embedding feature grids. |
| **[Cross-Attention Canvas (`dgem`)](../glossary.md#bidirectional-canvas-attention)** | Every input word directly inspects every policy rule and slot. | Zero-shot `AgentDrift` security audit & `LLM-AggreFact` grounding. |
| **[Normalized Entropy ($\tilde{H}_m$)](../glossary.md#cardinality-normalized-entropy)** | A universal `0.0–1.0` uncertainty gauge adjusted for option count. | Early-exiting a confident 26-way intent while escalating a torn 3-way NLI item. |
| **[Test-Time Compute (`think > 0`)](../glossary.md#fixed-depth-circuits-tc0-vs-test-time-compute)** | Scratchpad tokens generated only when a problem needs multi-step math. | Solving `2015 + 4 = 2019 > 2018` in `anli-02` (`--cascade-self-think 256`). |

<details class="term-aside">
<summary>💡 <strong>Concept Aside: Late Interaction (`GTR` Pooling Bottleneck) vs. Early All-to-All Cross-Attention (`dgem`)</strong> <em>(click to expand)</em></summary>

```text
1. GTR Dual-Encoder + TabPFN (Late Interaction / Vector Bottleneck):
   Input Text (1,000 tokens) ──► [T5 Encoder] ──► Single Vector u (R^768) ──┐
                                                                            ├──► [TabPFN Grid] ──► Prediction
   Policy Rules / Labels     ──► [T5 Encoder] ──► Label Vectors v_k       ──┘
   ⚠️ Bottleneck: Input tokens never attend to Policy tokens! Fine-grained numbers,
      negations ("NOT in allowlist"), and variable bindings are crushed during pooling.

2. DiffusionGemma Canvas Readout (Early Token-Level Cross-Attention):
   [Policy Rules (.json.tmpl) + Input Text (1,000 tokens) + Masked Slots <s_1, s_2, s_3>]
                                       │
              ┌────────────────────────┴────────────────────────┐
              │ All 26B-A4B Layers: Every token in Input,       │
              │ Policy, and Slots <s_1 <-> s_2> mutually attend │
              └────────────────────────┬────────────────────────┘
                                       ▼
                     Joint Calibrated Readout (one pass)
```

* **Why Normalization + `TabPFN` Cannot Undo Pooling Loss**: By the Data Processing Inequality, once $E_x(x)$ compresses a multi-clause passage or tool trajectory into a fixed vector $u \in \mathbb{R}^d$, information about which specific quantifier modifies which entity is lost. `TabPFN` is a powerful Bayesian decision boundary estimator over tabular columns, but it can only partition the features it is given—and it requires **in-context labeled support rows ($X_{\text{train}}, y_{\text{train}}$)**, whereas `dgem` executes declarative `.json.tmpl` policies with **zero support rows ($N_{\text{support}} = 0$)**.

</details>

<details class="term-aside">
<summary>💡 <strong>Concept Aside: Why Fixed-Depth Circuits (`think=0`) Cannot Solve Latent Multi-Hop Arithmetic Without Test-Time Compute (`think > 0`)</strong> <em>(click to expand)</em></summary>

* **The Circuit-Depth Bound ($\mathsf{TC}^0$)**: Any single forward pass through a transformer of fixed depth $L$ (`GTR`, `DeBERTa`, `TabPFN`, or `dgemma` at `steps=1, think=0`) executes a constant number of sequential layer operations.
* **Concrete Proof (`anli-02` in `dgem bench-calibration`)**:
  * **Premise**: *"Mira joined the lab in **2015** and became its second director **four years later**, succeeding the founder."*
  * **Hypothesis**: *"Mira led the lab before **2018**."*
  * Notice that the number **`2019`** never appears in the input tokens! To recognize the contradiction, the model must (1) bind `2015` + `four years later`, (2) compute the latent sum `2019`, and (3) evaluate `2019 < 2018` (`False` $\implies$ `contradiction`).
* **Why `dgem` Can Handle This Without Slowing Down Easy Traffic**: In Pass 1 (`think=0`), the model may output `entailment`, **but its normalized entropy $\tilde{H}_m$ rises above the gate ($\ge 0.160$)**. That triggers Pass 2 (`think > 0` with the Pass-1 prior block), which can compute `2015 + 4 = 2019` on its scratchpad and answer `contradiction`.

</details>

### Architectural Comparison Matrix

| Capability / Dimension | `GTR Dual-Encoder` + `TabPFN / TabFM` | Fine-Tuned Cross-Encoder (`DeBERTa-v3`) | `dgem` Single-Pass Canvas (`steps=1, think=0`) | `dgem` Prior-Guided Cascade (`think=0 → think>0`) |
| :--- | :--- | :--- | :--- | :--- |
| **Token-to-Policy Cross-Attention** | ❌ **No** (Late pooling into $u \in \mathbb{R}^d$) | ✅ **Yes** (Full cross-attention) | ✅ **Yes** (Full 26B-A4B cross-attention) | ✅ **Yes** (Full 26B-A4B cross-attention) |
| **Zero-Shot Policy Onboarding** | ⚠️ **Partial** (`TabPFN` requires $N>0$ support rows) | ❌ **No** (Requires fine-tuning per head) | ✅ **Yes** (via `.json.tmpl`, $N=0$ rows) | ✅ **Yes** (via `.json.tmpl`, $N=0$ rows) |
| **Joint Multi-Slot Readout (`s_1 <-> s_2`)** | ❌ **No** (1 target column per pass) | ❌ **No** (Independent linear heads) | ✅ **Yes** (`boolean` + `choice` + `score` in 1 pass) | ✅ **Yes** (`boolean` + `choice` + `score` in 1 pass) |
| **1-Hop Relational Grounding** | ⚠️ **Brittle** (Pooling loses parameter/negation scope) | ✅ **Strong** (If fine-tuned on domain) | ✅ **Strong** | ✅ **Strong** |
| **Latent Multi-Hop Arithmetic** | ❌ **Fails** (Fixed circuit depth, no scratchpad) | ❌ **Fails** (Fixed circuit depth, no scratchpad) | ❌ **Fails** (Single-pass $\mathsf{TC}^0$ limit) | ✅ **Possible** ($\tilde{H}_m \ge 0.16$ triggers `think>0` + Priors) |
| **Latency** | Lowest (small encoders) | Low | One forward pass of a large model | One pass, plus a reasoning pass for escalated items only |

---

## 6. Terminology: Discrete Diffusion Slot Readout

The formal technique is **Discrete Diffusion Slot Readout** (or bidirectional masked logit extraction). It builds directly upon foundational literature:
* **Masked Language Modeling (BERT, 2018)**: Evaluating logits across bidirectional transformer encoder layers.
* **Non-Autoregressive Sequence Generation (Mask-Predict, 2019)**: Parallel canvas denoising.
* **Discrete Denoising Diffusion (D3PM, 2021; MDLM, 2024)**: Denoising categorical state spaces.
* **[DiffusionGemma](https://gemma.ai/models/diffusiongemma/) (Google DeepMind, 2025/2026)**: The 26B-A4B MoE architecture providing an autoregressive prefix encoder paired with a discrete diffusion decoder with bidirectional attention.

`dgem` uses stock, unmodified weights from Google DeepMind (`google/diffusiongemma-26B-A4B-it`) and implements this technique directly.
