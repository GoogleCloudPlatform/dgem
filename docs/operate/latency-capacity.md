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
| `ENFORCE_EAGER` | 1 | Eager mode: shorter start-up, no graph compilation |

Set them as environment variables on the container; `scripts/deploy_vertex_endpoint.sh` and
`scripts/deploy_cloudrun_vllm.sh` pass the ones above. Change one at a time and measure before and after.

Keep-alive matters under load: the server speaks HTTP/1.1 with a 256-connection backlog. Use clients that reuse
connections.

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
