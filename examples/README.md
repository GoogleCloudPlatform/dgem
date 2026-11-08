# `dgem` Examples

Standalone, runnable Go and Python examples showing how to embed **DiffusionGemma (`dgem`)** and its executable `.json.tmpl` decision policies (`templates/`) into applications and agent frameworks.

Go examples share a single module (`examples/go.mod`, using `replace github.com/GoogleCloudPlatform/dgem => ..`) so extra framework dependencies are not added to the core `dgem` CLI binary, while Python examples (`agy-safety-gate/`) use the Python standard library only.

## Available Examples

| Example | Framework | Templates Used | What It Shows |
| :--- | :--- | :--- | :--- |
| **[`genkit-tool-approval/`](genkit-tool-approval/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/systemone/exp`, `plugins/middleware`, `ai/tool`) | `templates/support_triage.json.tmpl`<br/>`templates/calibration/agent_step_drift.json.tmpl` | 1. Executing existing `dgem` `.json.tmpl` policies through Genkit's `systemonex.SystemOne` plugin (`POST /v1/systemone`) with wall-time latency (`wall_ms`) and **Hesitation %** ($H / \ln K$) gating.<br/>2. Handling Genkit `v1.14.0` typed tool interrupts (`middleware.ToolApprovalInterrupted` and `genkit.DefineResumableTool`) where `confirm(name, input)` runs `agent_step_drift.json.tmpl` to block prompt-injected tool calls and resume approved ones via `ai.WithResume(...)`. |
| **[`genkit-bulk-synth/`](genkit-bulk-synth/)** | **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** (`plugins/googlegenai`, `plugins/systemone/exp`) | `templates/github_spam_triage.json.tmpl`<br/>`templates/contract_audit.json.tmpl` | **2-Stage Synthesize (`vertexai/gemini-3.8-flash`) $\to$ Bulk Decide (`dgem`)** pipeline: uses `genkit.GenerateData` with `vertexai/gemini-3.8-flash` (or a curated offline batch via `-curated`) to synthesize labelled test cases, then fans them out across a worker pool against `dgem`, reporting per-row latency, `min`/`p50`/`p95`/`max`, cold-vs-warm KV prefix cache latency, per-slot accuracy, and Hesitation-gated accuracy (`Clear <16%` vs. `Unsure >=16%`). |
| **[`agy-safety-gate/`](agy-safety-gate/)** | **[Google Antigravity](https://antigravity.google/)** (`agy` CLI [`PreToolUse` Hooks](https://antigravity.google/docs/hooks/) & Python SDK [`lifecycle`](https://antigravity.google/docs/sdk/lifecycle/) / [`policies`](https://antigravity.google/docs/sdk/policies/)) | `templates/tool_safety_gate.json.tmpl`<br/>`templates/calibration/agent_step_drift.json.tmpl` | Configurable `PreToolUse` safety gate (`auto_approve` / `ask` / `deny`) for `agy` CLI (`.agents/hooks.json`) and the `google.antigravity` Python SDK (`@hooks.pre_tool_call_decide` & `policy.ask_user` handler). Evaluates 2 orthogonal slots (`action_hazard` + `intent_origin`) in 1 pass with `use`-vs-`mention` awareness, `<untrusted_data>` prompt-injection quarantine, and **Hesitation %** ($H / \ln K < 16\%$) Stage 2 Gemini escalation. |
| **[`dgemgenkit/`](dgemgenkit/)** | Shared helper package | Any `templates/*.json.tmpl` | `EndpointConfig`, `NewSystemOnePlugin`, `DecideWithTemplate` & `StartInProcessSystemOneAdapter`: connects Genkit's `systemonex.SystemOne` plugin either directly to a `dgem serve` gateway / remote `/v1/systemone` endpoint (`DGEM_GATEWAY_URL`) or via an in-process adapter in front of a raw `/v1` endpoint (`DGEM_URL`), renders `.json.tmpl` files with `dgem/pkg/template`, builds `systemonex.Schema` questions (`NoulQuestion`, `ChoiceQuestion`, `ScoreQuestion`), calls `genkit.GenerateData`, and computes `WallTimeMs`, `EntropyNats(slot)`, and `Hesitation(slot)`. |

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
