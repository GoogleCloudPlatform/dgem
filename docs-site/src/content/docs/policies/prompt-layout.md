---
title: Prompt Layout (document first or questions first)
description: How dgem lays out the prompt for a decision, what changed in v0.2.0, the effect of each layout, when to choose the other one, and how to compare them on your own data.
---

Every decision sends the model two things: your **input** (the ticket, document, log or diff) and your **questions**
(each question with its allowed answers). The prompt layout decides the order and where each part goes.

| Layout | What the model sees | Default |
| :--- | :--- | :--- |
| `document_first` | One user message: your input inside `<user_text>`, then the questions inside `<instructions>` | **Yes, from serving v0.2.0** |
| `schema_first` | The questions as the system message, your input as the user message | Before v0.2.0 |

The questions, options, answer format and readout are identical in both layouts. Only the order and the message roles
change, so the model reads your input before or after it knows what it will be asked.

## What each layout looks like

For a policy with one question, `intent`, and the input `My card was charged twice for the same coffee.`:

**`document_first`** (one user message):

```text
<user_text>
My card was charged twice for the same coffee.
</user_text>

<instructions>
Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers;
reply with exactly one label per question.

Question intent: What does the customer want?
  A: refund (money back for a charge)
  B: new_card (a replacement card)
  C: balance (account balance)

Reply with one line per question, in this order, formatted as "id: label".
</instructions>
```

**`schema_first`** (system message, then user message):

```text
[system]  Answer a fixed set of questions ... Question intent: What does the customer want? ... "id: label".
[user]    My card was charged twice for the same coffee.
```

## Why document first is the default

We compared both layouts item by item on development data before changing the default, and again at the release
gate. In our tests `document_first` was more accurate on multi-question decisions (typed decisions with 5 questions
each), multilingual inference (XNLI), wide-option intent sets with an "out of scope" option (CLINC150) and
unsupported-content checks (RAGTruth). It made little or no difference on single, simple questions (JevBench,
MASSIVE intents).

The gain is largest when a decision has **several questions** or **many options**. With the questions first, the
model tends to settle on an answer shape before it has read the input. Single, simple questions barely change.

## Trade-offs to know about

- **"None of these" options are chosen less often.** In-scope requests are wrongly answered "out of scope" much less
  often, but genuinely out-of-scope requests are also recognised less often. If catching out-of-scope inputs matters
  more than classifying in-scope ones, compare both layouts on your data.
- **A few more prompt tokens.** The tags add a handful of tokens. A prompt right at the server's context limit can tip
  over and be refused. The deploy scripts default to 8,192 tokens on RTX PRO 6000, and the regression matrix reports
  refusals by reason.
- **Calibration** improved on most sets in our tests and got slightly worse on some (MASSIVE). Fitting a temperature
  per policy on your own data matters more than the layout here; see
  [calibrate your policy](/dgem/confidence/calibrate-your-policy/).

## When to choose `schema_first`

Keep the default unless one of these applies, and then measure before switching:

1. **You validated a policy before v0.2.0** and need byte-identical prompts (for example, an audited workflow). Pin
   `schema_first` in the template.
2. **Out-of-scope recall is the priority** for a policy with a catch-all option (see the trade-off above).
3. **Your prompts sit at the context limit** and a few extra tokens cause refusals (or ask your operator to raise
   the context length).

## How to set it

The narrowest setting wins: request, then template, then deployment.

| Where | How |
| :--- | :--- |
| A template (policy) | Add `"layout": "schema_first"` next to `"questions"` in the schema |
| One request to the server | `"layout": "schema_first"` in the schema (chat) or the `/v1/systemone` body |
| `dgem decide` | `--layout schema_first` |
| Gateway `POST /api/decide` | JSON `"layout": "schema_first"`, header `X-DGem-Layout`, or `?layout=` |
| MCP `decide_policy`, `decide_custom_questions` | `"layout": "schema_first"` |
| Web Studio | **Prompt layout** menu in the evaluation panel (also used by the batch runner) |
| `/v1/systemone` adapter | `dgem systemone serve --prompt-layout schema_first` |
| A whole deployment | Container env `DEFAULT_LAYOUT=schema_first` |

Leaving the field empty uses the template's own `layout`, or else the server's default. Responses report the layout the
server used in `diagnostics.layout`; the Studio shows it next to the menu. Servers older than v0.2.0 ignore the field
and always use `schema_first`.

## Yes/no questions in small requests are read alone

Asked in one joint read, a yes/no question that follows another one can copy its answer, mostly in small requests and
in agent or tool-call states. From serving v0.3.0 the server handles this automatically: in a **text** request with
**2 or 3 questions, at least 2 of them yes/no**, each yes/no question gets its own read, run in parallel within the
request with a prompt that lists only that question. Everything else is read jointly as before.

In our tests reading the yes/no questions alone clearly improved the second question in 2-question requests,
especially in agent and tool-call settings, and made no clear difference once the pair sat among more questions.

Requests outside the rule (one question, four or more questions, images) are unchanged, so latency and throughput on
them are unchanged. Isolating yes/no questions in every request (`"isolate": "noul"`) adds server time, lowers
throughput and slightly lowered accuracy on image decisions in our tests, so it stays opt-in.

**Controls:** `"isolate"` on a request or template: `"auto"` (default), `"noul"` (every yes/no question, requests of 2–8
questions), `"none"` (one joint read, the behaviour before v0.3.0) or `"all"`. `"alone": true` / `false` on a question
overrides it; `DEFAULT_ISOLATE` sets the deployment default. Responses list isolated questions in
`diagnostics.isolated`.

## How to compare both on your own data

Run the same labelled rows twice and compare accuracy, agreement and hesitation:

- **Studio:** load the dataset in the batch runner, run once per **Prompt layout** setting, and compare the results.
- **Python or CI:** add `"layout": "document_first"` or `"schema_first"` to the `POST /api/decide` body in the
  [dataset runner](/dgem/policies/datasets/) and run it twice.
- **Operators:** the [regression matrix](/dgem/operate/regression-matrix/) compares layouts on one deployment with
  `--target base=<URL> --target old=<URL>#layout=schema_first --baseline base`.

Use at least a few hundred labelled items, compare item by item, and decide before you look at the results what
difference would make you switch.
