# `dgem` Examples

Standalone, runnable Go examples showing how to embed **DiffusionGemma (`dgem`)** and its executable `.json.tmpl` decision policies (`templates/`) into applications and agent frameworks.

All examples share a single module (`examples/go.mod`, using `replace github.com/GoogleCloudPlatform/dgem => ..`) so extra framework dependencies are not added to the core `dgem` CLI binary.

## Available Examples

| Example | Framework | Templates Used | What It Shows |
| :--- | :--- | :--- | :--- |
| **[`genkit-tool-approval/`](genkit-tool-approval/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`, `plugins/middleware`, `ai/tool`) | `templates/support_triage.json.tmpl`<br/>`templates/calibration/agent_step_drift.json.tmpl` | 1. Executing existing `dgem` `.json.tmpl` policies through Genkit's `systemonex.SystemOne` plugin (`POST /v1/systemone`) with wall-time latency (`wall_ms`) and **Hesitation %** ($H / \ln K$) gating.<br/>2. Handling Genkit `v1.14.0` typed tool interrupts (`middleware.ToolApprovalInterrupted` and `genkit.DefineResumableTool`) where `confirm(name, input)` runs `agent_step_drift.json.tmpl` to block prompt-injected tool calls and resume approved ones via `ai.WithResume(...)`. |
| **[`genkit-bulk-synth/`](genkit-bulk-synth/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/googlegenai`, `plugins/systemone/exp`) | `templates/github_spam_triage.json.tmpl`<br/>`templates/contract_audit.json.tmpl` | **2-Stage Synthesize (`vertexai/gemini-3.8-flash`) $\to$ Bulk Decide (`dgem`)** pipeline: uses `genkit.GenerateData` with `vertexai/gemini-3.8-flash` (or a curated offline batch via `-curated`) to synthesize labelled test cases, then fans them out across a worker pool against `dgem`, reporting per-row latency, `min`/`p50`/`p95`/`max`, cold-vs-warm KV prefix cache latency, per-slot accuracy, and Hesitation-gated accuracy (`Clear <16%` vs. `Unsure >=16%`). |
| **[`dgemgenkit/`](dgemgenkit/)** | Shared helper package | Any `templates/*.json.tmpl` | `DecideWithTemplate` & `StartInProcessSystemOneAdapter`: renders `.json.tmpl` files with `dgem/pkg/template`, builds `systemonex.Schema` questions (`NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion`), calls `genkit.GenerateData`, and computes `WallTimeMs`, `EntropyNats(slot)`, and `Hesitation(slot)`. |

## Quick Start

### 1. Run All Offline Unit Tests (No GPU or Server Needed)

```bash
cd examples
go test -v ./...
```

### 2. Run Live Against Local Apple Silicon (`diffgemma`)

From the repository root:

```bash
# Start local diffgemma on 127.0.0.1:8080 (4096 context keeps memory footprint ~19.5 GiB on 32 GB Macs)
DIFFGEMMA_CTX=4096 ./scripts/serve.sh

# 1. Run the Genkit v1.14.0 tool-approval interrupt example
cd examples/genkit-tool-approval
go run .

# 2. Run the Genkit v1.14.0 bulk synthesis + evaluation example
cd ../genkit-bulk-synth
go run . -curated -count 6 -workers 1

# Stop the local diffgemma server when finished
cd ../..
make stop
```
