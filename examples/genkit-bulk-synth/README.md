# Genkit Go v1.14.0 Synthesize (`gemini-3.8-flash`) $\to$ Bulk `dgem` Policy Evaluation (`genkit-bulk-synth`)

This example demonstrates a two-stage **Synthesize $\to$ Bulk Decide** pipeline in **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)**:

1. **Stage 1 — Structured Test Synthesis (`vertexai/gemini-3.8-flash` via `genkit.GenerateData`)**:
   - Uses a generative model in Genkit (default `vertexai/gemini-3.8-flash`, or a built-in curated batch when running offline via `-curated`) to generate realistic items + expected ground-truth labels for a `dgem` `.json.tmpl` policy.
   - Records synthesis latency and `resp.TotalUsage.TotalTokens`.
2. **Stage 2 — Concurrent Bulk Evaluation (`systemonex.SystemOne` $\to$ `dgem`)**:
   - Evaluates the synthesized batch across a worker pool (`-workers`) using `dgemgenkit.DecideWithTemplate` against either:
     - **`templates/github_spam_triage.json.tmpl`** (`-preset github_spam_triage`, default): 2 simultaneous slots (`spam` boolean + 9-way `kind` choice across `genuine_bug`, `feature_request`, `question`, `docs_or_code_change`, `junk_or_test`, `low_effort_unfilled`, `off_topic`, `promotional_or_scam`, `malicious_or_abuse`).
     - **`templates/contract_audit.json.tmpl`** (`-preset contract_audit`): 3 simultaneous slots across all 3 System One question types (`auto_renews` boolean, `liability_cap` 3-way choice, `legal_risk` 4-level score).
   - Prints a per-row table and an aggregate scoreboard with:
     - **Total bulk wall time & throughput (`decisions/sec`)**
     - **Row #1 (cold prefill) vs. Rows #2..N (warm KV-cache reuse mean)**
     - **`min` / `p50` / `p95` / `max` latency**
     - **Per-slot & exact-match accuracy**
     - **Hesitation-Gated Accuracy** (`Clear <16%` vs. `Unsure >=16%` escalated to Stage 2 / human review).

## Running Offline Unit Tests (No GPU or Cloud Credentials Required)

```bash
cd examples/genkit-bulk-synth
go test -v ./...
```

## Running Live Against Local Apple Silicon Metal (`diffgemma`)

With `diffgemma serve` running on `:8080` (`DIFFGEMMA_CTX=4096 ./scripts/serve.sh` from repo root):

```bash
cd examples/genkit-bulk-synth

# 1. Run 6 GitHub spam/triage items on local M5 (workers=1 maximizes KV prefix cache reuse):
go run . -curated -count 6 -workers 1

# 2. Or synthesize fresh items with Vertex AI gemini-3.8-flash and evaluate on dgem:
GOOGLE_CLOUD_PROJECT=<PROJECT> go run . -synth-model vertexai/gemini-3.8-flash -count 10 -workers 1

# 3. Or run the 3-slot contract_audit.json.tmpl policy (boolean + choice + score):
go run . -preset contract_audit -curated
```
