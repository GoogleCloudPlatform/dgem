---
title: "Confidence and Calibration"
description: "How dgem assures confidence in its decisions: per-answer probabilities from one forward pass, hesitation, calibration checks, escalation, and what is recommended today."
---

# Confidence and Calibration

For teams who need to know when a `dgem` decision can be trusted, and how to check that. Illustrative numbers are
labelled as such. For the full picture of
why `dgem` decisions can be trusted, start with [Confidence beyond Shannon: hesitation-gated decisions](overview.md).

## What you get with every answer

DiffusionGemma fills in all answer slots of a policy at once and returns, for each question, a **probability for
every allowed answer** (not just a label). From those probabilities `dgem` derives:

| Signal | What it is | Where you see it |
| :--- | :--- | :--- |
| Confidence | Probability of the chosen answer | CLI table, JSON `confidence` |
| **Hesitation** | Entropy of the answer distribution scaled to 0–100% by the number of options (0% = certain, 100% = coin toss) | Decision Studio answer cards; `--stats` |
| Standard error, agreement | Spread across samples (with `samples: 4`) | CLI table, JSON |

Hesitation bands used in the Studio: **below 16% clear**, **16–50% somewhat unsure**, **above 50% very unsure**.
Why a model can read its own confidence in one pass, in plain language:
[The journey to decision models](../decision-models-primer.md) and the [glossary](../glossary.md).

## How we assure it

1. **Measure calibration on labelled data.** Confidence should match accuracy. Check it on public suites (for
   example JevBench, which `dgem bench-jev` runs, and the calibration suite in `dgem bench-calibration`) and on your
   own data: [Calibrate your policy](calibrate-your-policy.md).
2. **Gate on hesitation.** Answers above a hesitation threshold go to a person or a larger model. In our tests
   hesitation alone flags most errors.
3. **Escalate only what needs it.** A Stage 2 cascade to Gemini on hesitant answers recovers most of the larger
   model's accuracy while escalating a minority of items. Measure the escalation rate on your own traffic.
4. **Monitor known biases.** The model has a measurable preference for the first option when unsure; it is measured
   on every release rather than corrected, because the corrections we tested did not help reliably
   ([Confidence beyond Shannon](overview.md#43-order-bias-measured-and-monitored)).
5. **Re-validate on every serving change.** A new serving image is compared with your current deployment in the
   [regression matrix](../operate/regression-matrix.md), judged against the run-to-run agreement measured in the same
   session, before it takes traffic.

## What is recommended today

| Technique | Status | Notes |
| :--- | :--- | :--- |
| Hesitation gate (~16%) | **Recommended** | Catches most errors at low cost; fit the threshold on your data |
| Stage 2 cascade to Gemini on hesitant answers | **Recommended** where a larger model is acceptable | `cascade_mode: entropy` |
| `samples: 4` for an error bar | Use where agreement matters | One parallel batch; adds a little latency |
| Temperature scaling | **Recommended per policy**, fit on held-out data | The best temperature differs by domain |
| `document_first` prompt layout | **Default** from serving v0.2.0 | Helps most on multi-question and wide-option sets ([prompt layout](../policies/prompt-layout.md)) |
| Null-prior de-biasing | Not recommended | Helped some suites and hurt others in our tests |
| Reversed-order ("mirror") check | Not recommended | Little gain over hesitation for double the cost in our tests |

## Caveats

- **Small suites are noisy.** Suites of a few dozen or a few hundred items give coarse calibration estimates, and two
  identical runs do not agree on every answer ([regression matrix](../operate/regression-matrix.md)).
- **Individual answers depend on the serving image.** A runtime upgrade can leave accuracy unchanged while some items
  change answer; don't compare per-item results across images.
- **Calibration is policy- and data-specific.** Numbers on public suites don't transfer automatically; check your
  own ([how](calibrate-your-policy.md)).

## Read more

- [Confidence beyond Shannon: hesitation-gated decisions](overview.md): the techniques behind trustworthy decisions.
- [How the model decides in one pass](architecture.md): discrete diffusion vs autoregression.
- [Regression matrix](../operate/regression-matrix.md): how serving changes are checked.
