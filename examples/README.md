# `dgem` Examples

Standalone, runnable Go and Python examples showing how to embed **DiffusionGemma (`dgem`)** and its executable `.json.tmpl` decision policies (`templates/`) into applications and agent frameworks.

Go examples share a single module (`examples/go.mod`, using `replace github.com/GoogleCloudPlatform/dgem => ..`) so extra framework dependencies are not added to the core `dgem` CLI binary, while Python examples (`agy-safety-gate/`) use the Python standard library only.

## Available Examples

| Example | Framework | Templates Used | What It Shows |
| :--- | :--- | :--- | :--- |
| **[`genkit-tool-approval/`](genkit-tool-approval/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`, `plugins/middleware`, `ai/tool`) | `templates/support_triage.json.tmpl`<br/>`templates/calibration/agent_step_drift.json.tmpl` | 1. Executing existing `dgem` `.json.tmpl` policies through Genkit's `systemonex.SystemOne` plugin (`POST /v1/systemone`) with wall-time latency (`wall_ms`) and **Hesitation %** ($H / \ln K$) gating.<br/>2. Handling Genkit `v1.14.0` typed tool interrupts (`middleware.ToolApprovalInterrupted` and `genkit.DefineResumableTool`) where `confirm(name, input)` runs `agent_step_drift.json.tmpl` to block prompt-injected tool calls and resume approved ones via `ai.WithResume(...)`. |
| **[`genkit-secops-triage/`](genkit-secops-triage/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`, `plugins/middleware`, `ai/tool`) | `templates/secops_conditional_dag.json.tmpl`<br/>`templates/secops_rag_quarantine.json.tmpl`<br/>`templates/secops_iam_change_audit.json.tmpl` | 1. **2-Stage Conditional DAG Alert Response** (`depends_on` / `ask_if`, running Stage-2 containment & blast-radius slots only when `active_breach == "yes"`).<br/>2. **RAG `Use`-vs-`Mention` Indirect Prompt-Injection Quarantine** (`allow_in_context` vs. `quote_as_inert_data` vs. `quarantine_and_alert`).<br/>3. **IAM / Terraform Diff Gating** combining `middleware.ToolApproval` (`auto_approve` / `block_change`) with resumable `tool.Interrupt` human SecOps review (`require_secops_review`). |
| **[`genkit-rai-guardrails/`](genkit-rai-guardrails/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`, `plugins/googlegenai`) | `templates/rai/prompt_safety_guardrail.json.tmpl`<br/>`templates/rai/custom_content_policy.json.tmpl`<br/>`templates/rai/response_safety_grader.json.tmpl`<br/>`templates/rai/unbiased_context_qa.json.tmpl`<br/>`templates/rai/provenance_claim_check.json.tmpl` | 1. **Calibrated `(T, \tau)` Input Guardrail** (`strict` / `balanced` / `permissive` operating points with benign figurative/homonym over-refusal protection).<br/>2. **Post-Calibration Hesitation-Gated Stage-2 Gemini Cascade** (computing $H_{\text{norm}}$ *after* temperature scaling before escalating to `vertexai/gemini-3.8-flash` or human review).<br/>3. **Single-Pass Red-Team Transcript Grading**, **Unbiased Context QA Abstention (`cannot_be_determined`)**, and **C2PA Content Credentials Provenance Check**. |
| **[`genkit-bulk-synth/`](genkit-bulk-synth/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/googlegenai`, `plugins/systemone/exp`) | `templates/github_spam_triage.json.tmpl`<br/>`templates/contract_audit.json.tmpl`<br/>`templates/security_incident.json.tmpl`<br/>`templates/rai/response_safety_grader.json.tmpl` | **2-Stage Synthesize (`vertexai/gemini-3.8-flash`) $\to$ Bulk Decide (`dgem`)** pipeline across `-preset github_spam_triage`, `contract_audit`, `secops_incident`, and `rai_redteam_grader`: synthesizes labelled test batches (or runs curated offline batches via `-curated`), fans them out across a worker pool against `dgem`, and reports `min`/`p50`/`p95`/`max` latency, cold-vs-warm KV cache speedup, per-slot accuracy, and Hesitation-gated accuracy (`Clear <16%` vs. `Unsure >=16%`). |
| **[`genkit-context-pruning/`](genkit-context-pruning/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`) | `templates/agent_context_pruning.json.tmpl` | **Single-Pass 8-Slot Trajectory Context Pruning**: scores up to 8 candidate trajectory blocks (`b0`..`b7`) across `full` / `stub` / `drop` in one `dgem` pass and solves the exact Pareto-filtered Multiple-Choice Knapsack under a hard `TokenBudget` and `Pinned` block constraints. |
| **[`genkit-consistency-arbiters/`](genkit-consistency-arbiters/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`) | `templates/multimodal/entity_resolution_dedup.json.tmpl`<br/>`templates/swarm_lock_arbiter.json.tmpl`<br/>`templates/live_checklist_verify.json.tmpl` | Combines single-pass `dgem` readout with deterministic post-decision arbiters: **Entity Resolution Pack-Quantity & 3-Cycle Transitivity Guardrails** (`ValidatePackQuantityConsistency`, `ScoreProbabilisticTriangles`), **4-Worker Swarm Lock Contention Arbitration** (`ArbitrateSwarmLocks` with `< 16%` hesitation gating), and **Sliding-Window Operational Checklist Verification**. |
| **[`agy-safety-gate/`](agy-safety-gate/)** | **[Google Antigravity](https://antigravity.google/)** (`agy` CLI [`PreToolUse` Hooks](https://antigravity.google/docs/hooks/) & Python SDK [`lifecycle`](https://antigravity.google/docs/sdk/lifecycle/) / [`policies`](https://antigravity.google/docs/sdk/policies/)) | `templates/tool_safety_gate.json.tmpl`<br/>`templates/calibration/agent_step_drift.json.tmpl` | Configurable `PreToolUse` safety gate (`auto_approve` / `ask` / `deny`) for `agy` CLI (`.agents/hooks.json`) and the `google.antigravity` Python SDK (`@hooks.pre_tool_call_decide` & `policy.ask_user` handler). Evaluates 2 orthogonal slots (`action_hazard` + `intent_origin`) in 1 pass with `use`-vs-`mention` awareness, `<untrusted_data>` prompt-injection quarantine, and **Hesitation %** ($H / \ln K < 16\%$) Stage 2 Gemini escalation. |
| **[`dgemgenkit/`](dgemgenkit/)** | Shared helper package | Any `templates/*.json.tmpl` | `EndpointConfig`, `NewSystemOnePlugin`, `DecideWithTemplate` & `StartInProcessSystemOneAdapter`: connects Genkit's `systemonex.SystemOne` plugin directly to a `dgem serve` gateway / remote `/v1/systemone` endpoint (`DGEM_GATEWAY_URL`) or via an in-process adapter (`DGEM_URL`), executes multi-stage conditional DAG policies (`depends_on` / `ask_if`, `Passes`, `Skipped`), and provides post-hoc calibration helpers (`ApplyTemperature`, `RawHesitation`, `CalibratedProb`, `ExceedsThreshold`, `CalibratedHesitation`). |

## Endpoint Configuration (`dgemgenkit.EndpointConfig`)

All examples share `dgemgenkit.EndpointConfigFromEnv()` and `dgemgenkit.BindEndpointFlags`, so the same code and compiled binaries switch seamlessly between local Metal, an Apple `container` microVM bridge (`192.168.64.1`), or a remote hosted Cloud Run / Vertex AI endpoint:

| Environment Variable | CLI Flag | Description |
| :--- | :--- | :--- |
| `DGEM_GATEWAY_URL` (or `DGEM_SYSTEMONE_URL`) | `-gateway-url` | Direct `dgem serve` gateway or `/v1/systemone` URL (e.g. `http://127.0.0.1:8095`, `http://192.168.64.1:8095` from an Apple `container` microVM, or `https://<gateway>.a.run.app`). Calls `POST /v1/systemone` directly over HTTP(S). |
| `DGEM_URL` | `-url` | Upstream `/v1` endpoint (default `http://127.0.0.1:8080/v1`, automatically wrapped by the in-process `/v1/systemone` adapter) or a direct gateway URL. |
| `DGEM_MODEL` | `-model` | Model identifier (default `diffgemma-26b-a4b-it-q4`). |
| `DGEM_BACKEND` | `-backend` | Optional `X-DGem-Backend` header for gateway routing (`vertex_first`, `vertex`, `cloudrun`, `local`). |
| `DGEM_AUTH_TOKEN` | — | Optional Bearer / IAM / OIDC token sent in `Authorization: Bearer <token>`. |
| `DGEM_GATEWAY_KEY` | — | Optional `X-DGem-Api-Key` sent to the gateway matching `DGEM_GATEWAY_URL`. |
| `DGEM_SERVER_KEY` | — | Optional `X-DGem-Key` sent to a serving image. |

## Quick Start

### 1. Run All Offline Unit Tests (No GPU or Server Needed)

```bash
cd examples
go test -v ./...
python3 -m unittest -v agy-safety-gate/test_dgem_gate.py
```

### 2. Run Live Against Local Apple Silicon (`diffgemma`) or a Gateway

From the repository root:

```bash
# Option A: Start local diffgemma on 127.0.0.1:8080 (4096 context keeps memory footprint ~19.5 GiB on 32 GB Macs)
DIFFGEMMA_CTX=4096 ./scripts/serve.sh

# Option B: Point all examples at a running dgem serve gateway instead:
# export DGEM_GATEWAY_URL=http://127.0.0.1:8095

# 1. Run the Genkit v1.14.0 tool-approval interrupt example
go -C examples/genkit-tool-approval run .

# 2. Run the Genkit v1.14.0 SecOps conditional DAG, RAG quarantine & IAM gate example
go -C examples/genkit-secops-triage run .

# 3. Run the Genkit v1.14.0 calibrated RAI & safety guardrails example
go -C examples/genkit-rai-guardrails run .

# 4. Run the Genkit v1.14.0 bulk synthesis + evaluation example across presets
go -C examples/genkit-bulk-synth run . -preset github_spam_triage -curated
go -C examples/genkit-bulk-synth run . -preset secops_incident -curated
go -C examples/genkit-bulk-synth run . -preset rai_redteam_grader -curated

# Stop the local diffgemma server when finished (if started in Option A)
make stop
```
