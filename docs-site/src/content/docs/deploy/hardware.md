---
title: "Hardware and GPU Sizing"
description: "Measured GPU configurations for DiffusionGemma serving: RTX PRO 6000, H100 80GB, A100 80GB, and L4 — required settings, NVFP4 vs BF16 vs FP8 weights, and latency/throughput ranges."
---

This page summarizes measured single-GPU configurations for the `dgem` serving image on Vertex AI dedicated endpoints,
along with recommended settings for Cloud Run, GKE, and self-hosted hosts.

**Short answer:** use an **NVIDIA RTX PRO 6000 (Blackwell, ~96 GB VRAM)** with the default **NVFP4** checkpoint. It is
faster and lower cost per decision than every other GPU we measured—including an H100 80GB—because Blackwell executes
4-bit FP4 MoE layers natively (`FLASHINFER_CUTLASS`), whereas Hopper and Ampere fall back to weight-only dequantization
kernels (`MARLIN`) or move ~2.7× more weight bytes in BF16.

## Recommended configurations by GPU

| GPU (architecture, VRAM) | Vertex machine (host RAM) | Recommended weights | `DISABLE_MM` | `KV_CACHE_GB` | `MAX_SEQS` | `MAX_INFLIGHT` | `MAX_MODEL_LEN` | `ENFORCE_EAGER` | Required `EXTRA_ARGS` / notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX PRO 6000** (Blackwell, ~96 GB) | `g4-standard-48` (180 GB) | **NVFP4** *(default)*, FP8, or BF16 | `0` (vision on) | `12` | `32` | `8` | `8192` | `1` *(default)* or `0` | **Recommended.** Native FP4 and FP8 kernels. BF16 (~48.6 GiB loaded) also fits on one card in eager mode (`ENFORCE_EAGER=1`). |
| **H100 80GB** (Hopper, 80 GB) | `a3-highgpu-1g` (208 GB) | **BF16** or **FP8** (or NVFP4) | `0` (vision on) | `12` | `32` | `8` | `8192` | `0` (for latency) or `1` | NVFP4 runs via weight-only Marlin fallback (not fixed-seed reproducible). BF16 and FP8 run native Hopper kernels. |
| **A100 80GB** (Ampere SM80, 80 GB) | `a2-ultragpu-1g` (170 GB) | NVFP4 | `0` (vision on) | `12` | `32` | `8` | `8192` | `0` or `1` | **Required:** `EXTRA_ARGS="--kv-cache-dtype bfloat16"`. Default FP8 KV cache requires SM89+ and crash-loops on A100. |
| **L4** (Ada, 24 GB) | `g2-standard-16` (64 GB) or `g2-standard-8` (32 GB) | NVFP4 | `1` (text only) | `2` | `16`–`32` | `8` | `4096` | **`1` (required)** | Keep `ENFORCE_EAGER=1` (CUDA graphs OOM on first request at 2 GB KV). On `g2-standard-8`, `DISABLE_MM=1` is required to fit host RAM. |

Platform notes:
- **Vertex AI:** pass `VERTEX_PROFILE=g4-rtxpro6000|h100|a100|l4` to `scripts/deploy_vertex_endpoint.sh`
  ([Production on Vertex AI](/dgem/deploy/vertex/#2-deploy)). The `a100` profile sets `EXTRA_ARGS="--kv-cache-dtype bfloat16"`
  automatically.
- **Cloud Run GPU:** supports 1× RTX PRO 6000 (~96 GB VRAM, 80 GiB host RAM) and 1× L4 (24 GB VRAM, 32 GiB host RAM
  with `DISABLE_MM=1`). Scale-to-zero; same serving image ([Deploy on Cloud Run](/dgem/deploy/cloud-run/)).
- **GKE / self-hosted NVIDIA host:** use the same container environment variables (`DISABLE_MM`, `KV_CACHE_GB`,
  `MAX_SEQS`, `MAX_MODEL_LEN`, `ENFORCE_EAGER`, `EXTRA_ARGS`); mount weights locally or stage from Cloud Storage
  ([Build the serving image](/dgem/deploy/public-images/#6-running-on-your-own-gpu-host)).

> **Readiness check:** always wait for `GET /ready` HTTP `200` (or `"vllm_ready": true` in `/health`) before routing
> traffic. `GET /health` always returns HTTP `200` so cold-start progress stays readable even if vLLM fails to start.

## Measured latency and throughput ranges

**Method:** single-GPU Vertex AI dedicated endpoints, 1 replica per endpoint, identical serving image and pinned
weights per weight format, same-region client job, short structured-decision latency suite (single-sample server-side
median `diagnostics.timing.total_ms` and concurrency-sweep peak throughput). Accuracy on the text regression suites was
unchanged across GPUs within run-to-run noise.

| GPU | Weights | Eager (`ENFORCE_EAGER=1`) server p50 / peak req/s | CUDA graphs (`ENFORCE_EAGER=0`) server p50 / peak req/s | Fixed-seed reproducibility (`seed_token: "fixed"`) |
| :--- | :--- | :--- | :--- | :--- |
| **RTX PRO 6000** (~96 GB) | NVFP4 | **~55 ms** / ~70–75 req/s | **~26 ms** / ~112–118 req/s | Reproducible in eager mode; **not** reproducible with CUDA graphs |
| **RTX PRO 6000** (~96 GB) | FP8 | — | **~27 ms** / ~109 req/s | Reproducible with CUDA graphs (lower sustained mixed-traffic capacity than NVFP4 eager; see below) |
| **RTX PRO 6000** (~96 GB) | BF16 | **~50 ms** / ~70 req/s | Fails startup (compile timeout) | Reproducible in eager mode |
| **H100 80GB** | NVFP4 | ~111 ms / ~39 req/s | ~31 ms / ~89 req/s | Lower reproducibility (Marlin fallback) |
| **H100 80GB** | BF16 | — | ~35 ms / ~82 req/s | Reproducible with CUDA graphs |
| **H100 80GB** | FP8 | ~123 ms | ~28 ms | Reproducible on both GPUs (check wide-option accuracy on your policies) |
| **A100 80GB** (bf16 KV) | NVFP4 | ~172 ms / ~25 req/s | ~58 ms / ~52 req/s | — |
| **L4** (24 GB, text only) | NVFP4 | ~174 ms / ~25 req/s | Fails (OOM on first request) | — |

Key takeaways:
- **RTX PRO 6000 vs H100:** the RTX PRO 6000 was faster in every measured configuration (~2.0× faster like-for-like in
  eager mode, ~1.2× faster with NVFP4 + CUDA graphs, ~1.3× faster than H100 BF16 + CUDA graphs, and ~1.1× faster than
  H100 FP8 + CUDA graphs) with **~2.6–5.5× lower cost per 1,000 decisions** at on-demand list prices.
- **FP8 on RTX PRO 6000:** with a public FP8 checkpoint and `ENFORCE_EAGER=0`, the RTX PRO 6000 matches NVFP4 +
  CUDA graphs on single-sample latency (~27 ms, ~109 req/s) while keeping answers bit-reproducible under
  `seed_token: "fixed"`. Under a mixed production traffic workload (multi-question, vision, and long states), FP8 +
  CUDA graphs saved ~12 ms server median (~51 vs ~63 ms) but lowered sustained throughput (~20 vs ~30 req/s) compared
  to NVFP4 in eager mode, so the default remains NVFP4 with `ENFORCE_EAGER=1`.
- **Unquantized BF16 on RTX PRO 6000:** because the Vertex `g4-standard-48` RTX PRO 6000 exposes ~94.4 GiB free VRAM
  (not 48 GB), unquantized BF16 (~48.6 GiB loaded) plus a 12 GB KV cache fits on a single card. In eager mode it was
  ~11% faster than NVFP4 (~50 vs ~57 ms server p50) with the same text accuracy and fixed-seed reproducibility, though
  `ENFORCE_EAGER=0` did not finish startup compile within the platform window.
- **Eager vs CUDA graphs trade-off:** see [Latency and capacity — Eager vs CUDA graphs](/dgem/operate/latency-capacity/#eager-vs-cuda-graphs-enforce_eager).

## Cloud TPU status

Cloud TPU is **not recommended for self-hosted `dgem` serving today**. Two independent BF16 text-only paths were
evaluated:

| TPU configuration | Stack | Measured behavior | Status |
| :--- | :--- | :--- | :--- |
| **Cloud TPU v5e ×8** (`v5litepod-8`, 128 GB HBM) | Google JAX (`gemma/diffusion`, text-only, batch 1) | ~715–730 ms device time per read (~1,180 ms at 4k tokens); ~22× slower than RTX PRO 6000 FP8 + CUDA graphs | Not recommended |
| **Cloud TPU v6e ×4** (`v6e-4`, 128 GB HBM) | `vllm-torchtpu` (BF16, TP=4) | Default runner rejects diffusion models; V2 runner loads but fails on the first decision (`TPUModelRunner` lacks V2 `ModelState` hooks) | Cannot serve without an upstream runner port |

## What was not measured and multi-GPU limits

- **Untested GPUs:** H200 (8-GPU machine shape only on Vertex serving) and B200 / other datacenter Blackwell cards have
  not been measured; run the [evaluation checklist](/dgem/deploy/evaluate/) before relying on them.
- **Tensor parallelism (`TP > 1`):** vLLM's constrained structured-decision readout (`diffusion_constrained=True`)
  currently requires `tensor_parallel_size=1`. To scale throughput beyond one GPU, run **multiple 1-GPU replicas**
  (`VERTEX_MAX_REPLICAS > 1`) behind an endpoint or gateway rather than sharding one replica across GPUs.

## Tip: mirror images into Artifact Registry without local Docker pulls

If you mirror a prebuilt or registry image into another region or project's Artifact Registry, you can copy layers
directly in Cloud Build with `crane` (~20 s, no local Docker download):

```bash
gcloud builds submit --no-source --project=<PROJECT> \
  --config=<(cat <<'EOF'
steps:
  - name: gcr.io/go-containerregistry/crane:latest
    args: ["copy", "<SOURCE_IMAGE_URI>", "<DEST_IMAGE_URI>"]
EOF
)
```
