# Genkit Go v1.14.0 Synthesize (`gemini-3.8-flash`) $\to$ Bulk `dgem` Policy Evaluation (`genkit-bulk-synth`)

This example demonstrates a two-stage **Synthesize $\to$ Bulk Decide** pipeline in **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)**:

1. **Stage 1 — Structured Test Synthesis (`vertexai/gemini-3.8-flash` via `genkit.GenerateData`)**:
   - Uses a generative model in Genkit (default `vertexai/gemini-3.8-flash`, or a built-in curated batch when running offline via `-curated`) to generate realistic items + expected ground-truth labels for a `dgem` `.json.tmpl` policy.
   - Records synthesis latency and `resp.TotalUsage.TotalTokens`.
2. **Stage 2 — Concurrent Bulk Evaluation (`systemonex.SystemOne` $\to$ `dgem`)**:
   - Evaluates the synthesized batch across a worker pool (`-workers`) using `dgemgenkit.DecideWithTemplate` against four built-in presets:
     - **`templates/github_spam_triage.json.tmpl`** (`-preset github_spam_triage`, default): 2 simultaneous slots (`spam` boolean + 9-way `kind` choice).
     - **`templates/contract_audit.json.tmpl`** (`-preset contract_audit`): 3 simultaneous slots across all 3 System One question types (`auto_renews` boolean, `liability_cap` 3-way choice, `legal_risk` 4-level score).
     - **`templates/security_incident.json.tmpl`** (`-preset secops_incident`): 3 simultaneous SecOps triage slots (`data_compromise` boolean, `action` 4-way choice, `severity` 4-level score `p3_low`..`p0_critical`).
     - **`templates/rai/response_safety_grader.json.tmpl`** (`-preset rai_redteam_grader`): 4 simultaneous red-team / output-safety grading slots (`prompt_adversarial`, `response_harmful`, `response_refusal`, and 7-way `harm_category`).
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

## Running Live

With a `dgem serve` gateway (`-gateway-url "$DGEM_GATEWAY_URL"`) or `diffgemma serve` running on `:8080`:

```bash
cd examples/genkit-bulk-synth

# 1. Run 6 GitHub spam/triage items (workers=1 maximizes local KV prefix cache reuse):
go run . -curated -count 6 -workers 1

# 2. Synthesize fresh items with Vertex AI gemini-3.8-flash and evaluate on dgem:
GOOGLE_CLOUD_PROJECT=<PROJECT> go run . -synth-model vertexai/gemini-3.8-flash -count 10 -workers 4

# 3. Run SecOps incident triage or RAI red-team response grading batches:
go run . -preset secops_incident -curated
go run . -preset rai_redteam_grader -curated

# 4. Run the 3-slot contract_audit.json.tmpl policy (boolean + choice + score):
go run . -preset contract_audit -curated
```
