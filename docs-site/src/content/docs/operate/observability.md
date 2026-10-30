---
title: "OpenTelemetry Trace Waterfall & Cloud Observability Guide"
description: "How to analyze and interpret dgem.gateway.decide, dgem.gpu.orchestrate, dgem.gpu.forward_pass, dgem.gpu.prefill, dgem.gpu.denoise, and dgem.gpu.cold_start_backoff spans across Google Cloud Trace, Cloud Logging, and Cloud Monitoring."
---

Every decision request handled by **`dgemma-gateway`** (`POST /api/decide`, `POST /api/decide/{template}`, and the Decision Studio Web UI) is instrumented with **OpenTelemetry** (`cmd/otel.go` and `cmd/serve.go`) and exported to **Google Cloud Trace**, **Cloud Run Structured Logging**, and the gateway's built-in **Trace Ring Buffer (`GET /api/traces`)**.

MCP decide tools (`decide_policy`, `decide_custom_questions`) are traced differently. They record the `dgem.gpu.orchestrate` / `dgem.gpu.forward_pass` spans (and `dgem.taxonomy.expand` when used), but not the `dgem.gateway.decide` root span, so their traces have no gateway root and they don't return `trace_id` or `trace_spans`. Their responses do include the same `max_entropy`, `gpu_forward_ms`, and `cold_start_wait_ms` summary fields as `/api/decide`.

This guide explains what `dgem.gateway.decide` and `dgem.gpu.orchestrate` measure, why they exhibit two very different latency regimes (cold start vs. warm forward pass), how child spans segment each GPU pass into `network_and_auth` $\to$ `prefill` $\to$ `denoise`, and where to inspect them in Google Cloud Observability.

---

## 1. The OpenTelemetry Span Waterfall

When a decision evaluation executes, `dgemma-gateway` records a 4-level hierarchical Gantt waterfall ordered top-down (`Depth 0` $\to$ `Depth 3`) with microsecond-accurate `offset_ms`:

```mermaid
flowchart TD
    Root["<b>[Depth 0] dgem.gateway.decide</b><br/>Root HTTP Request Wrapper<br/>(total_wall_ms = template_render + cold_start_wait_ms + gpu_forward_ms)"]
    Render["<b>[Depth 1] dgem.template.render</b><br/>Go .json.tmpl Compilation & Schema Check<br/>(offset_ms: +0.0ms · duration_ms &lt; 1 ms)"]
    Orch["<b>[Depth 1] dgem.gpu.orchestrate</b><br/>Scale-from-Zero Absorb & Retry Loop<br/>(warmup_attempts, cold_start_wait_ms, gpu_forward_ms)"]
    Pass1["<b>[Depth 2] dgem.gpu.forward_pass [attempt=1]</b><br/>Upstream HTTP POST to dgemma<br/>(Status=Error 502/ConnectionRefused if cold)"]
    Backoff["<b>[Depth 2] dgem.gpu.cold_start_backoff</b><br/>6s Retry Backoff While GPU Container Boots"]
    PassN["<b>[Depth 2] dgem.gpu.forward_pass [attempt=N]</b><br/>Winning GPU Forward Pass Wrapper<br/>(Status=Ok · gpu_forward_ms)"]
    Net["<b>[Depth 3] dgem.gpu.network_and_auth</b><br/>IAM Token + HTTP/TLS Transport"]
    Prefill["<b>[Depth 3] dgem.gpu.prefill</b><br/>Prompt Tokenization + SigLIP Vision Tower"]
    Denoise["<b>[Depth 3] dgem.gpu.denoise</b><br/>Bidirectional Diffusion Canvas Denoising (reads, steps)"]

    Root --> Render
    Root --> Orch
    Orch --> Pass1
    Orch --> Backoff
    Orch --> PassN
    PassN --> Net
    PassN --> Prefill
    PassN --> Denoise
```

```text
dgem.gateway.decide (depth=0, total_wall_ms, gpu_forward_ms, cold_start_wait_ms, max_entropy)
 ├── dgem.template.render (depth=1, template.id, template.path, duration_ms < 1ms)
 └── dgem.gpu.orchestrate (depth=1, upstream_url, backend, warmup_attempts)
      ├── dgem.gpu.forward_pass [attempt=1, status=Error]      <-- Only during cold boot
      ├── dgem.gpu.cold_start_backoff [attempt=1, 6000ms]      <-- Only during cold boot
      └── dgem.gpu.forward_pass [attempt=N, status=Ok] (depth=2, gpu.forward_ms)
           ├── dgem.gpu.network_and_auth (depth=3, gpu.network_ms)
           ├── dgem.gpu.prefill (depth=3, gpu.prefill_ms, prompt_tokens, image_count)
           └── dgem.gpu.denoise (depth=3, gpu.denoise_ms, reads, steps)
```

### Span Reference Table

| Span Name | Depth | Role & What It Measures | Key Attributes |
| :--- | :---: | :--- | :--- |
| **`dgem.gateway.decide`** | `0` | **Root HTTP Request Wrapper.** Wraps the complete lifecycle of a decision evaluation from incoming HTTP request to JSON response serialization. | `dgem.total_wall_ms`, `dgem.gpu.forward_ms`, `dgem.gpu.prefill_ms`, `dgem.gpu.denoise_ms`, `dgem.gpu.cold_start_wait_ms`, `dgem.gpu.reads`, `dgem.max_entropy`, `dgem.template`, `dgem.surface`, `dgem.user` |
| **`dgem.template.render`** | `1` | **Policy Compilation (Leaf).** Renders the `.json.tmpl` policy with caller variables and validates the decision slot schema (`< 1 ms`). | `dgem.template.id`, `dgem.template.path` |
| **`dgem.gpu.orchestrate`** | `1` | **Serverless GPU Wakeup & Retry Wrapper.** Absorbs Cloud Run scale-from-zero (`0 -> 1` GPU instance) by retrying upstream calls every `6 seconds` until `vLLM` is ready. | `dgem.warmup_attempts`, `dgem.gpu.cold_start_wait_ms`, `dgem.gpu.forward_ms`, `dgem.upstream_url`, `dgem.backend`, `dgem.model` |
| **`dgem.gpu.cold_start_backoff`** | `2` | **Scale-from-Zero Retry Wait (Leaf).** Explicit `6s` backoff interval between cold-start probe attempts while `vLLM` streams safetensors into GPU VRAM. | `dgem.attempt`, `dgem.reason` |
| **`dgem.gpu.forward_pass`** | `2` | **Single Upstream GPU Attempt Wrapper.** Represents one HTTP call to `dgemma`. Contains the 3 contiguous execution leaf spans (`network_and_auth`, `prefill`, `denoise`) when `Status=Ok`. | `dgem.attempt`, `dgem.gpu.forward_ms`, `dgem.gpu.network_ms`, `dgem.gpu.prefill_ms`, `dgem.gpu.denoise_ms`, `dgem.gpu.reads`, `dgem.gpu.steps` |
| **`dgem.gpu.network_and_auth`** | `3` | **Transport & Auth Overhead (Leaf).** IAM/Bearer token header injection, TLS handshake, and JSON payload serialization. | `dgem.gpu.network_ms`, `dgem.backend` |
| **`dgem.gpu.prefill`** | `3` | **Prompt & Vision Encoding (Leaf).** Causal prompt token prefill (`dgem.gpu.prompt_tokens`) and `SigLIP` vision tower feature extraction (`dgem.image_count`). | `dgem.gpu.prefill_ms`, `dgem.gpu.prompt_tokens`, `dgem.image_count` |
| **`dgem.gpu.denoise`** | `3` | **Bidirectional Canvas Denoising (Leaf).** Iterative discrete diffusion denoising across all decision slots simultaneously (`dgem.gpu.steps` and `dgem.gpu.reads`). | `dgem.gpu.denoise_ms`, `dgem.gpu.reads`, `dgem.gpu.steps` |
| **`dgem.gpu.warmup_lifecycle`** | `0` | **4-Stage GPU Boot Telemetry.** Emitted whenever a background scale-from-zero wakeup completes, decomposing total boot time across the 4 hardware initialization stages. | `dgem.warmup.total_ms`, `dgem.warmup.stage1_activator_ms`, `dgem.warmup.stage2_tmpfs_ms`, `dgem.warmup.stage3_vllm_siglip_ms`, `dgem.warmup.stage4_triton_jit_ms` |

---

## 2. Why Do `dgem.gateway.decide` and `dgem.gpu.orchestrate` Have Similar Inclusive Widths?

In OpenTelemetry, parent wrapper spans record **inclusive wall-clock duration** (`EndTime - StartTime`). Because `dgem.template.render` takes `< 1 ms`, the root HTTP span (`dgem.gateway.decide`) and its orchestration child (`dgem.gpu.orchestrate`) naturally span almost the same time window.

In the **Decision Studio Gantt Waterfall**, parent wrapper spans (`gateway.decide`, `gpu.orchestrate`, `gpu.forward_pass`) are rendered as **dashed outline container brackets**, while the **active execution leaf spans** (`template.render`, `cold_start_backoff`, `network_and_auth`, `prefill`, `denoise`) are rendered as **solid color-coded Gantt bars positioned at their exact `left: offset%` and `width: duration%` coordinates**.

High total durations fall into one of **two distinct regimes**:

### Regime A: Serverless Cold Start (`total_wall_ms` of minutes)

To cost nothing while idle, the upstream `dgemma` Cloud Run GPU service scales down to `0` instances (`--min-instances=0`). When a request arrives while scaled to zero:

1. **Port 8080 (`structured_server.py`) Starts Before Port 8000 (`vLLM`)**:
   The lightweight Python reverse proxy passes Cloud Run startup probes immediately so the container activates, while `vLLM`'s `EngineCore` takes a few minutes to load weights in the background ([Deploy on Cloud Run §7](/dgem/deploy/cloud-run/#7-cold-start-what-to-expect-and-your-options)) (`ConnectionRefusedError(111)` on port 8000).
2. **The 4 Hardware Boot Stages** (recorded on `dgem.gpu.warmup_lifecycle`):
   * **Stage 1 — Cloud Run GPU Activator (`stage1_activator_ms`)**: GPU VM allocation and container scheduling.
   * **Stage 2 — Cloud Storage $\to$ in-memory staging (`stage2_tmpfs_ms`)**: Copying **17.53 GiB** of NVFP4 safetensors into RAM (several times faster with Direct VPC egress than over public egress).
   * **Stage 3 — `vLLM` `EngineCore` & `SigLIP` Init (`stage3_vllm_siglip_ms`)**: Loading model weights into GPU VRAM and initializing Gemma 4's `SigLIP` vision encoder.
   * **Stage 4 — First-Inference Triton JIT Compilation (`stage4_triton_jit_ms`)**: Compiling custom `TRITON_ATTN` kernels on the first forward pass.
3. **Visible `dgem.gpu.cold_start_backoff` Spans**:
   Rather than failing the caller's HTTP request with `502 Bad Gateway`, `dgem.gpu.orchestrate` holds the request open and emits `dgem.gpu.cold_start_backoff` spans (`6,000 ms` each) between retries until `vLLM` answers.

### Regime B: Warm GPU Inference

When the GPU is already warm (`dgem.warmup_attempts = 1` and `dgem.gpu.cold_start_wait_ms = 0`), a 3-question decision takes tens of milliseconds of GPU time on an RTX PRO 6000; end to end adds network and routing ([Latency and capacity](/dgem/operate/latency-capacity/)). Three things make a decision slower:

1. **Eager Execution (`--enforce-eager`) & `TRITON_ATTN`**:
   FlashAttention-2 and CUDA Graphs reject DiffusionGemma's mixed **causal-prompt + bidirectional-canvas** attention mask. Running `TRITON_ATTN` in eager mode introduces Python/kernel dispatch overhead across the iterative denoising steps (`dgem.gpu.steps`, typically 8–16 steps per read).
2. **2-Stage Conditional Policy DAGs (`dgem.gpu.reads = 2`)**:
   Whenever a `.json.tmpl` policy uses `depends_on` / `ask_if` (for example, `secops_conditional_dag.json.tmpl`), `structured_server.py` partitions the questions into `level 0` and `level 1` in `schedule(qs)`, executing **2 sequential GPU forward passes (`reads=2`)** and doubling `dgem.gpu.denoise`.
3. **Multimodal `SigLIP` Vision Encoding (`dgem.multimodal = true`)**:
   Passing an image or bounding-box canvas runs the `SigLIP` vision tower and multi-scale visual token projection during `dgem.gpu.prefill` prior to canvas denoising (`dgem.gpu.denoise`).

---

## 3. Where to View Traces in Google Cloud Observability

`initGatewayTracer` (`cmd/otel.go`) exports telemetry to four surfaces simultaneously:

### A. Google Cloud Trace (Visual Waterfall)
* **Console URL**: `https://console.cloud.google.com/traces/list?project=<PROJECT>`
* Filter by service name **`dgemma-gateway`** or span name **`dgem.gateway.decide`**.
* Every `/api/decide` response returns an HTTP header **`X-Dgem-Trace-Id`** and JSON property **`trace_id`**. Paste this 32-character hex ID into Cloud Trace to inspect the correlated `dgemma-gateway` $\to$ `dgemma` span waterfall.

### B. Google Cloud Logging Explorer (Correlated Structured Logs)
Every completed span emits a structured JSON log entry with `logging.googleapis.com/trace` and `logging.googleapis.com/spanId` fields so logs and traces cross-link automatically in the GCP Console.

* **Query 1 — All Decision Evaluations (with Prefill, Denoise, Latency, Reads & Entropy):**
  ```text
  resource.type="cloud_run_revision"
  jsonPayload.span_name="dgem.gateway.decide"
  ```
* **Query 2 — Warm GPU Decisions Only (Exclude Scale-from-Zero Cold Starts):**
  ```text
  resource.type="cloud_run_revision"
  jsonPayload.span_name="dgem.gateway.decide"
  jsonPayload.dgem_cold_start_wait_ms = 0
  ```
* **Query 3 — Cold-Start 4-Stage Hardware Warmup Breakdowns:**
  ```text
  resource.type="cloud_run_revision"
  jsonPayload.span_name="dgem.gpu.warmup_lifecycle"
  ```

### C. Google Cloud Monitoring Operational Dashboard & Log-Based Metrics
Provisioned via `scripts/setup_cloud_monitoring.sh` (`deploy/monitoring/dgem-operational-dashboard.json`):
* **Dashboard**: **DiffusionGemma (`dgem`) — Operational & Decision Intelligence Dashboard**
* **Custom Metrics (`logging.googleapis.com/user/*`)**:
  * `dgem_decisions_total` — Decision request volume broken down by `surface`, `template`, `user`, `multimodal`, and `reads`.
  * `dgem_gpu_forward_ms` — Distribution of pure GPU forward-pass latency (`ms`), cleanly excluding cold-start wait time.
  * `dgem_gpu_prefill_ms` — Distribution of GPU prompt + `SigLIP` vision prefill duration (`ms`).
  * `dgem_gpu_denoise_ms` — Distribution of bidirectional diffusion canvas denoising duration (`ms`).
  * `dgem_epistemic_entropy` — Distribution of calibrated Shannon entropy ($H_{\max}$ in nats) by `template` and `surface`.
  * `dgem_warmup_total_ms` — Distribution of scale-from-zero GPU warmup durations by trigger source.

### D. Gateway Ring Buffer API (`GET /api/traces`) & Decision Studio UI
* **Decision Studio Web UI**: Directly underneath the slot readout cards, the **OpenTelemetry Request & GPU Model Span Waterfall (Gantt Timeline)** panel renders hierarchical tree rows with dashed parent wrappers and solid leaf bars positioned at `[+offset_ms, duration_ms]`.
* **REST API (`GET /api/traces`)**:
  ```bash
  # Fetch the exact hierarchical span waterfall for a specific trace_id
  curl -s -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    "${GATEWAY_URL}/api/traces?trace_id=${TRACE_ID}" | jq .
  ```

## 4. Request and Response Body Logging (Off by Default)

The gateway's normal logs carry the caller, route, status, error reason and request shape (question counts, image
count), never request or response content. When you need to see what a caller sent and what it got back (a client
that behaves differently after a release, say), turn on body logging for a limited time and for named callers only.

**What it does.** For each matching request on `/v1/systemone`, `/api/decide` or `/mcp` (`tools/call` only), the
gateway writes one extra JSON line to stdout with `"log_type": "dgem.bodies"`:

| Field | Content |
| :--- | :--- |
| `dgem_user`, `route`, `http_status`, `trace_id` | Who called, which route, the status returned, the trace (also set as `logging.googleapis.com/trace`) |
| `request` | The request body (only with `DGEM_LOG_BODIES=both`) |
| `response`, `response_bytes` | The response body and its full size |
| `request_truncated`, `response_truncated` | `true` when a body was longer than 96 KB and was cut (Cloud Logging entries hold at most 256 KB) |
| `logged_until` | When logging switches itself off |

Images are never logged: every `data:` URI and every base64 string of 512+ characters, anywhere in a body, is
replaced by `{"redacted": "data_uri" or "base64", "mime", "chars", "sha256"}` (the first 16 hex digits of the
digest), so you can still tell whether two requests carried the same image. Multipart uploads and binary bodies are
recorded by type and size only. Headers, tokens and other routes (`/api/auth/*`, admin routes) are never logged.

**Settings** (environment variables on the gateway service):

| Variable | Required | Meaning |
| :--- | :---: | :--- |
| `DGEM_LOG_BODIES` | yes | `off` (default), `responses` (responses only) or `both` |
| `DGEM_LOG_BODIES_UNTIL` | yes | RFC3339 time, at most 7 days ahead; nothing is logged after it, so the setting can't be left on by mistake |
| `DGEM_LOG_BODIES_USERS` | yes | Comma-separated caller identities as they appear in `dgem.user` (an IAP account, `apikey:<id>`); write `*` to log every caller |
| `DGEM_LOG_BODIES_SAMPLE` | no | Fraction of matching requests to log, `(0, 1]`, default `1` |

An invalid combination (missing or past `UNTIL`, more than 7 days ahead, no callers) leaves logging off. The gateway
prints the reason at startup, and `GET /health` reports it under `body_logging`:

```json
"body_logging": {"enabled": true, "mode": "both", "until": "2026-10-22T18:00:00Z", "callers": 1, "sample": 1}
```

**Keep the content out of your normal logs.** Bodies are your callers' data. Route them to a separate log bucket with
short retention and narrow read access, and exclude them from `_Default`, before you turn logging on:

```bash
PROJECT=<PROJECT>; REGION=<REGION>
gcloud logging buckets create dgem-bodies --project=$PROJECT --location=$REGION --retention-days=7
gcloud logging sinks create dgem-bodies \
  "logging.googleapis.com/projects/$PROJECT/locations/$REGION/buckets/dgem-bodies" --project=$PROJECT \
  --log-filter='resource.type="cloud_run_revision" AND jsonPayload.log_type="dgem.bodies"'
gcloud logging sinks update _Default --project=$PROJECT \
  --add-exclusion=name=dgem-bodies,filter='jsonPayload.log_type="dgem.bodies"'
# Read access: grant roles/logging.viewAccessor on a log view of the dgem-bodies bucket to the people who debug.
```

**Turn it on and off.** Setting the variables deploys a new Cloud Run revision. Deploy it with `--no-traffic --tag`
first and check `/health`, then move traffic. Logging stops by itself at `UNTIL`; to stop earlier, remove
`DGEM_LOG_BODIES` (another revision). Read entries with:

```text
jsonPayload.log_type="dgem.bodies" AND jsonPayload.dgem_user="<caller>"
```
