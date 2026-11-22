---
title: "Latency and Capacity"
description: "What drives dgem latency and throughput: request design (samples, questions per request, thinking, isolation), engine settings (MAX_INFLIGHT, MAX_SEQS, KV_CACHE_GB, MAX_MODEL_LEN), and how to measure and plan capacity for your own deployment."
---

# Latency and Capacity

Latency and throughput depend on your policies, your GPU, your concurrency and where your clients run, so plan
capacity from measurements on your own deployment. This page covers what moves the numbers, which knobs exist, and
how to measure. GPU time below means the server-reported denoise time (`diagnostics.timing.total_ms`, shown as
**Server Denoise** by `dgem decide --stats`); end to end adds network, routing and any gateway.

## Request design: the biggest levers

- **`samples: 1` by default.** One read is the fastest path. Use `samples: 4` (one parallel batch, a modest
  increase in GPU time) where you need agreement and standard error, for example to route to human review. Avoid
  `"auto"` on latency-sensitive paths: it usually runs a second sequential batch. Schemas without `samples` use the
  server's `DEFAULT_SAMPLES` (1).
- **Ask everything about one input in one request.** All questions share one forward pass; three questions cost
  about the same as one.
- **Avoid `depends_on` / `ask_if` when latency matters.** Conditional questions run in a second stage (two
  sequential passes). Gate in your client instead if you need a single pass.
- **At most 26 options per `choice` question.** Wider taxonomies need a two-stage split (two passes).
- **Thinking is the most expensive option.** A `think` budget adds a reasoning trace whose cost grows with its
  length (hundreds of milliseconds for ~120 tokens on an RTX PRO 6000). Use it only where it measurably helps (for
  example taxonomy discovery).
- **Isolation adds parallel reads.** Questions read alone (`"isolate"` on the request, `"alone"` on a question, or
  the server's `DEFAULT_ISOLATE`) each get an extra read in the same batch. The default `auto` only isolates yes/no
  questions in small text requests (2–3 questions); `noul` (every yes/no question) costs noticeably more latency
  and throughput on larger requests. See
  [reading yes/no questions alone](../policies/prompt-layout.md#yesno-questions-in-small-requests-are-read-alone).
- **Images and long inputs** add prefill time; keep the per-row `state` short and keep instructions and options
  fixed so the prefix cache is reused.

## Engine and server settings

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `MAX_INFLIGHT` | 8 | Decisions processed at once per server; the rest wait up to `INFLIGHT_WAIT_S` (30 s), then `503`. Protects the engine from overload: without a cap, too many concurrent multi-sample requests can crash it. |
| `DEFAULT_SAMPLES` | 1 | Samples for schemas that don't set `samples` |
| `MAX_SEQS` | 32 | vLLM sequences in one batch (a 4-sample request uses 4) |
| `KV_CACHE_GB` | 12 (Vertex G4 profile), 2 (Cloud Run script, image default) | KV cache memory; more allows more concurrent sequences and longer prompts |
| `MAX_MODEL_LEN` | 4096 (image); 8192 in the deploy scripts on RTX PRO 6000 | Longest prompt plus answer, in tokens. Longer prompts are refused, so raise it if your inputs are long; it needs KV cache memory |
| `DEFAULT_ISOLATE` | `auto` | Which questions get a read of their own when a request doesn't say (`auto`, `none`, `noul`, `all`) |
| `DISABLE_MM` | 0 | `1` turns off the vision tower (required on L4 Cloud Run, saves memory) |
| `ENFORCE_EAGER` | 1 | `1` = eager mode (faster start-up, fixed-seed reproducible on NVFP4). `0` = CUDA graphs + `torch.compile` (lower latency and higher throughput, not fixed-seed reproducible on NVFP4; see [below](#eager-vs-cuda-graphs-enforce_eager)). |

Set them as environment variables on the container; `scripts/deploy_vertex_endpoint.sh` and
`scripts/deploy_cloudrun_vllm.sh` pass the ones above. Change one at a time and measure before and after.

### Eager vs CUDA graphs (`ENFORCE_EAGER`)

Setting `ENFORCE_EAGER=0` enables CUDA graphs and `torch.compile` with `TRITON_ATTN`. On a single RTX PRO 6000
serving the NVFP4 checkpoint, switching only `ENFORCE_EAGER` from `1` to `0` measured:

- **Latency:** ~2.2× lower server-side median decision time (~55–58 ms → ~26 ms for a single question, with similar
  gains on 4-question and image requests).
- **Throughput:** ~1.5× higher peak throughput (~112–118 vs ~70–85 req/s at `samples: 1`).
- **Start-up:** adds ~1 minute of compile and graph capture at boot (~1 s on the first real request after start-up).
- **Determinism and answer shift:** with NVFP4 weights, `ENFORCE_EAGER=0` is **no longer reproducible** under
  `seed_token: "fixed"`, and a small fraction of items shift stably beyond run-to-run noise even though regression-suite
  accuracy stays unchanged.

Because of that determinism trade-off, the deploy scripts keep `ENFORCE_EAGER=1` by default. Set `ENFORCE_EAGER=0`
only when you have measured your own workload and do not require fixed-seed reproducibility on NVFP4 (and never on an
L4 at `KV_CACHE_GB=2`, where CUDA graphs OOM on the first request). See
[Hardware and GPU sizing](../deploy/hardware.md) for per-GPU measurements across RTX PRO 6000, H100, A100, and L4.

Keep-alive matters under load: the server speaks HTTP/1.1 with a 256-connection backlog. Use clients that reuse
connections.

### Size the canvas and concurrency to your policy

On sparse mixture-of-experts (MoE) GPUs (`26B` total parameters, `~4B` active per token across `8` of `128` routed experts), throughput peaks at moderate batch concurrency (typically `8`–`16` active sequences on a single 80 GB GPU) and drops if concurrency grows unbounded:

- **Size per-request `diffusion_canvas_length` and the `CANVAS` (`--canvas`) ceiling to your template.** Container and deploy scripts set the `CANVAS` environment variable to `128` by default (`32` on L4 Cloud Run), while standalone `structured_server.py` defaults `--canvas` (`CANVAS_LEN`) to `64`. For each request, `canvas_width(template)` (`min(CANVAS_LEN, -(-need // CANVAS_STEP) * CANVAS_STEP)` with `CANVAS_STEP = 16`) rounds the required token length (`need = len(template) + 1`) up to the smallest multiple of `16` (`16` for 1- and 2-question `lines` templates, `32` for 5 questions, and `64` for 10 questions), capped at `CANVAS_LEN`, rather than padding short requests out to the full `CANVAS_LEN` ceiling. Measured on the `google/diffusiongemma-26B-A4B-it` SentencePiece tokenizer, a single-question `lines` template (`<|channel>thought\n<channel|>allow: yes<turn|>`) occupies `8` active tokens (`7` template tokens plus `<turn|>`, or `9` active tokens for `q1: yes`), while `2`, `5`, and `10` boolean or single-token `choice` questions occupy `14`, `29`, and `55` active tokens. If you call raw vLLM directly without `structured_server.py`, set `diffusion_canvas_length` per request to the smallest multiple of `16` that holds the template (`16` for 1–2 questions instead of `64` or `128`), because trailing `PAD` (`0`) positions still participate in bidirectional canvas attention and MoE expert dispatch without changing decision accuracy. Set a lower `CANVAS` (or `--canvas`, for example `32` or `16`) when you want a tighter upper ceiling on deployments that never serve `think` traces or wide multi-question policies.
- **Cap `MAX_INFLIGHT` and `MAX_SEQS` at the MoE knee.** Past `16`–`32` simultaneous sequences (and earlier when `samples: 4` or `8` multiplies each decision into `4`–`8` vLLM sequences), ragged batches activate nearly all `128` experts per layer—turning a `4B`-active step into a `26B` memory-bandwidth step—while KV-cache pressure triggers chunked-prefill preemption. Set `MAX_INFLIGHT` (`8`–`16`) so burst traffic queues cleanly in userspace instead of thrashing the engine scheduler.
- **Set `DISABLE_MM=1` on text-only deployments.** Passing `--language-model-only` skips loading and profiling the vision tower, freeing GPU memory for `KV_CACHE_GB`.

## Scaling out

- **Vertex AI:** one dedicated-endpoint replica per GPU; set `VERTEX_MIN_REPLICAS` / `VERTEX_MAX_REPLICAS` and the
  endpoint autoscales on GPU duty cycle (`AUTOSCALE_DUTY_CYCLE`, default 70%). Add replicas for more throughput.
- **Cloud Run:** the deploy script runs one GPU instance per service (max instances 1, concurrency 32), so a single
  service's throughput is one GPU's. Use it for bursty and batch work, or put a [gateway](../deploy/gateway.md) in
  front that prefers a Vertex endpoint.

## Measure your own deployment

```bash
# Per-mode latency (repeat --target to compare deployments side by side)
python3 scripts/serving_speed.py modes --target mine=<BASE_URL>/v1/chat/completions -n 30 -o modes.json

# Concurrency sweep; any error is a finding
python3 scripts/serving_speed.py sweep --target mine=<BASE_URL>/v1/chat/completions \
  --workers 1 4 8 16 32 --samples 1 4 --min-requests 256 -o sweep.json

# Cloud Run cold start for one configuration
python3 scripts/coldstart_probe.py --service <SERVICE> --project <PROJECT> --region <REGION> --label mine --vpc
```

`<BASE_URL>` is a Cloud Run URL (`<CLOUD_RUN_URL>`) or a Vertex invoke base
(`https://<ENDPOINT_ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>/invoke`).
Tokens come from Application Default Credentials. `modes` covers 1 sample, 4 samples, `"auto"`, a
reversed-order check, a reasoning trace and an image request (`--modes` to pick); every request records wall time, server GPU time, reads and errors, so failures are never counted as
zero latency.

How to plan capacity from the results:

1. Run `modes` to get per-request GPU and end-to-end time for request shapes like yours. If your policies differ
   (more questions, images, `think`), time them with `dgem decide --stats` or a matrix latency run.
2. Run `sweep` up to and beyond your expected concurrency. Throughput grows with concurrency until requests start
   to queue (around `MAX_INFLIGHT` and the KV cache limit); past that point latency rises while throughput
   plateaus. That plateau is one replica's capacity for that request shape.
3. Divide your peak decisions per second by that capacity, add headroom, and set the replica count (Vertex) or
   decide how much traffic Cloud Run should take.
4. Re-measure after changing the image, an engine setting or the policies; the
   [regression matrix](regression-matrix.md) latency suite compares a candidate with your baseline in one session.

Measure from the region your clients run in, and don't run benchmarks against the same endpoint at the same time
(they contaminate each other's latency).
