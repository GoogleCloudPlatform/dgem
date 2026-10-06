---
title: "Confidence beyond Shannon: hesitation-gated decisions"
description: "Why you can trust a dgem decision: a probability for every allowed answer from one pass, a hesitation score that flags uncertain answers, hand-off to a larger model, and the prompt, calibration and evaluation practices behind them."
---

`dgem` turns DiffusionGemma into a **decision model**: you describe a decision as a few typed questions, and for every
question you get **a probability for every allowed answer** from a single forward pass. This page explains how those
probabilities become decisions you can trust: a **hesitation score** that flags uncertain answers, a **gate** that hands
them to a larger model or a person, and the prompt, calibration and evaluation practices that keep both honest.

> **In one sentence:** `dgem` answers clear cases itself in a single fast pass and knows when it is unsure, and you
> can measure that "knowing", tune it per domain, and re-check it on every release.

---

## 1. What every decision gives you

| You get | What it is | Where |
| :--- | :--- | :--- |
| **A probability for every option** | Not just a label: the full distribution over the answers you allowed | CLI JSON, HTTP gateway, MCP tools, Decision Studio |
| **Confidence** | Probability of the chosen answer | Same |
| **Hesitation** | How torn the model is, 0–100% (below) | Studio answer cards, `--stats`, JSON `entropy` |
| **Diagnostics** | Passes, layout used, latency, backend used | JSON `diagnostics`, `X-DGem-Backend-Used` |

All questions in a policy are answered **together in one pass**, so reading several questions costs about the same as
reading one. Reading each question in its own pass multiplies latency and, with the default prompt layout, does not
reliably improve accuracy. Measure the latency of your own deployment with the
[regression matrix](/dgem/operate/regression-matrix/).

---

## 2. The first step: Shannon entropy, shown as hesitation

From the answer probabilities `dgem` computes Shannon entropy and divides it by its maximum for that question, so a
2-option and a 26-option question share one 0–100% scale:

$$\text{hesitation} = \frac{-\sum_k p_k \ln p_k}{\ln K}$$

0% means all the probability is on one answer; 100% means a perfect tie. Decision Studio labels it **Clear** (below
16%), **Somewhat unsure** (16–50%) and **Very unsure** (above 50%), with the raw value in nats on hover.

This is an approachable first step for anyone assessing probabilities: no labelled data, no extra passes, and it
works on every question type. In our tests it is also a strong one: wrong answers have much higher hesitation than
right ones, so hesitation alone is a useful signal for which answers to double-check.

---

## 3. Hesitation-gating: the core technique

**Hesitation-gating** means: answer directly when hesitation is low, and hand the decision to a larger model (or a
person) when it is high, passing along `dgem`'s probabilities as a hint.

The fast path answers clear items in a single pass; only hesitant items pay for the larger model. In our tests a gate recovers most of the larger model's accuracy while handing off a minority of items. How
many items are handed off, and how much accuracy that buys, depends on your domain and threshold, so measure it on
your own data.

*Illustrative example (not a measurement):* if a gate at 25% hesitation hands off 1 item in 5, the larger model
handles 20% of your traffic and `dgem` answers the other 80% directly, each in one pass.

**Choosing a threshold.** Start at 16–35% and fit it on a labelled sample of your own decisions, choosing the
threshold on one half and checking it on the other ([Calibrate your policy](/dgem/confidence/calibrate-your-policy/)). The gate is
available on every surface (`cascade_mode`, `cascade_threshold`, `cascade_model`; see
[authoring and Stage 2 cascades](/dgem/policies/authoring/)). Note that the gateway's `cascade_threshold` is raw entropy
in nats (default 0.35), not hesitation %; multiply hesitation by ln K to convert (16% hesitation ≈ 0.18 nats for 3
options).

---

## 4. What makes the answers trustworthy

Hesitation is only useful if the probabilities behind it are well formed. These practices keep them that way.

### 4.1 The answer template is part of the prompt

The model reads the whole answer template, including question ids and option labels. Two effects follow:

- **Shared letters cause copying.** When two questions label different options with the same letters (A, B, C…),
  the model tends to copy the letter across questions. Labelling the second question with digits avoids this.
- **Question ids act as instructions.** A single question's id makes little difference, but naming a second question
  something like `…__mirror_rev` changes answers even with identical options.

**What you do:** use neutral, descriptive ids (`team`, `urgent`, `severity`) and avoid ids that hint the answer
should differ (`mirror`, `reverse`, `opposite`). This is a template rule ([templates](/dgem/policies/templates/)).

### 4.2 Prompt layout

Since serving v0.2.0 the default layout puts your input first and the questions after it (`document_first`). In our
tests this improved accuracy on multi-question decisions, multilingual inference and wide-option intent sets, and
made little difference on single questions ([prompt layout](/dgem/policies/prompt-layout/)).

**Trade-off:** with a catch-all "out of scope" option, out-of-scope recall is lower under `document_first`, and
calibration can shift. Layout can be set per template, per request, or per deployment on every surface.

### 4.3 Order bias: measured and monitored

When unsure, the model leans toward the first option listed, most strongly with few options. The regression
matrix's option-order suites shuffle option order and report the extra answer flips over an identical repeat, so you
can check order bias on every release.

In-pass corrections (dividing out a blank-question prior, or reading a reversed copy of the options) did not improve
decisions reliably in our tests, so order bias is **tracked as a release gate rather than corrected**. Hesitation
already catches most of the cases it affects.

### 4.4 Calibration by domain

Raw probabilities are somewhat over-confident, and by how much depends on the domain: the temperature that best
calibrates confidence differs noticeably between domains (for example, natural-language inference needs a much
higher temperature than short intent questions).

**What you do:** fit one temperature per policy on held-out decisions; a single global value does not fit every
domain ([Calibrate your policy](/dgem/confidence/calibrate-your-policy/)). Temperature never changes which answer wins, only how
confident it reports.

### 4.5 Results judged against measured noise

The model is not perfectly deterministic: two identical runs do not agree on every answer. The
[regression matrix](/dgem/operate/regression-matrix/) measures this agreement in the same session and judges a
candidate against it (paired McNemar tests), with pinned datasets and a receipt for every run. A result that is within
noise is reported as such.

---

## 5. Strong, flexible, explainable, transparent

| Property | What provides it |
| :--- | :--- |
| **Strong** | One fast pass per decision; hesitation-gating recovers most of a larger model's accuracy while handing off a minority of items (§3). |
| **Flexible** | Decisions are templates you edit, not models you retrain; any mix of yes/no, choice and score questions; per-template layout and temperature; four surfaces (CLI, HTTP, MCP, Studio). |
| **Explainable** | A probability for every option and a hesitation score on every answer; the reason an item was handed off is visible (its hesitation); diagnostics report layout and passes. |
| **Transparent** | Every run of the regression matrix writes receipts you can inspect; candidates are judged against noise measured in the same session. |

---

## 6. What it does not do (yet)

- **Missing context:** if a decisive fact is absent, the model can be confidently wrong. List required facts as their
  own question.
- **Rewording:** rephrasing options can shift answers.
- **Base rates:** confidence is not adjusted for how common each class is in your traffic.
- **Out-of-scope recall** is lower under the default layout (§4.2).
- **Images:** categorical image questions work, and hesitation flags wrong answers there too. `dgem`'s own bounding
  boxes are much less accurate than Gemini's; for boxes, use `dgem` to skip absent targets and give Gemini a grid-cell
  hint ([images guide](/dgem/policies/images/)).
- **Multi-step reasoning:** one pass cannot do arithmetic or long chains of inference; these items show up as hesitant
  and are handed off.

---

## 7. Prior art and what is new

Entropy-based abstention and confidence-gated cascades (selective prediction, Geifman & El-Yaniv 2017; FrugalGPT,
Chen et al. 2023) and temperature scaling (Guo et al. 2017) are established techniques; `dgem` applies them to a
single-pass slot readout. What is specific to `dgem`:

- **Joint reads at single-question cost:** many typed questions answered together, each with a full probability
  distribution, in one pass.
- **Template effects in slot readouts:** shared labels and loaded question ids change answers, turned into template
  rules.
- **Noise-judged release gates:** accuracy, calibration, coverage and option order compared against agreement
  measured in the same session, on every serving change.

---

## 8. Quick reference

```bash
# One decision, with hesitation and timing
./bin/dgem decide -t templates/support_triage.json.tmpl -v ticket="..." --stats

# Choose a prompt layout for a request (default: document_first)
./bin/dgem decide -t templates/support_triage.json.tmpl -v ticket="..." --layout schema_first

# Hesitation-gated hand-off to Gemini (HTTP gateway)
curl -s http://localhost:8090/api/decide/support_triage -H 'Content-Type: application/json' \
  -d '{"variables":{"ticket":"..."},"cascade_mode":"entropy","cascade_threshold":0.35}'   # threshold in nats

# Check a serving change against your current deployment, judged against measured noise
python3 scripts/bench_matrix.py run --tier T1 --target prod=<url> --target new=<url> --baseline prod
```

Related: [Confidence and calibration](/dgem/confidence/) (how to use the signals day to day) ·
[Calibrate your policy](/dgem/confidence/calibrate-your-policy/) · [Prompt layout](/dgem/policies/prompt-layout/) ·
[Regression matrix](/dgem/operate/regression-matrix/) · [Glossary](/dgem/glossary/).
