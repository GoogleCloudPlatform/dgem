---
title: "Step 1: Run on Your Laptop"
description: "Run DiffusionGemma locally on Apple Silicon (Metal) or a local NVIDIA GPU with Docker, build the dgem CLI, make a first decision, and open Decision Studio."
---

# Step 1: Run on Your Laptop

Get a first structured decision running on your own machine. This is the **crawl** stage of
[From Laptop to Production](index.md): ideal for writing policies, trying templates on private data, and
working offline.

## 1. Start an engine

Pick the option that matches your hardware.

### Option A: Mac with Apple Silicon (Metal), no Docker or cloud

```bash
make setup       # check prerequisites
make download    # 4-bit model pack, ~18.8 GB
make local-up    # diffgemma engine on :8080 + dgem gateway and Decision Studio on :8090
make local-status
```

A 32 GB Mac works with the recommended context size; memory budgeting, the engine's own API and troubleshooting
are in the [Apple Silicon engine reference](../reference/metal-engine.md). Stop everything with `make local-down`.

### Option B: Linux or Windows workstation with an NVIDIA GPU (Docker)

The serving image needs a GPU that can run the NVFP4 checkpoint (Blackwell, e.g. RTX PRO 6000, is what we test;
an L4 works but is slower; see [GPU requirements](evaluate.md#0-hardware)). There are no prebuilt images yet:
first [build the serving image](public-images.md) into your own Artifact Registry
(`GCP_PROJECT=<PROJECT> make cloudrun-build`), then run it:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

# Lean image: downloads the public weights from Hugging Face on first boot (needs internet)
docker run --gpus all -p 8080:8080 \
  us-central1-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag>

# Or mount a local copy of the weights instead of downloading them
docker run --gpus all -p 8080:8080 -v /data/dgemma:/weights -e MODEL=/weights \
  us-central1-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag>
```

Wait until
`curl -s localhost:8080/health` shows `"vllm_ready": true` (and `"warmed": true` a few seconds later).

## 2. Build the CLI

```bash
make build        # -> ./bin/dgem
```

## 3. Make your first decision

```bash
./bin/dgem decide \
  -t templates/support_triage.json.tmpl \
  -v 'ticket=Urgent: production database connection pool exhausted after deploy' \
  --stats
```

Example output (illustrative, not a recorded run):

```text
SLOT             | TYPE       | VALUE                | PROB       | ENTROPY     | MARGIN
-----------------------------------------------------------------------------------------
sentiment        | score      | frustrated           | 99.8%      | 0.002 nats  | 1.00
team             | choice     | engineering          | 100.0%     | 0.000 nats  | 1.00
urgent           | boolean    | yes                  | 99.9%      | 0.001 nats  | 1.00
```

All three questions are answered in **one forward pass**, each with its own probabilities. `support_triage`
opts in to 4 samples (one parallel batch) so it can report agreement; most templates use one. How to read
probabilities and hesitation: [Your first decision policy](../policies/first-policy.md).

## 4. Open Decision Studio

`make local-up` already started it; otherwise run `./bin/dgem serve --port 8090` and open
<http://localhost:8090>. Browse the built-in policies, run one on your own text or image, and inspect the
per-question probabilities and the trace waterfall.

## 5. Connect an AI assistant (MCP)

```json
{
  "mcpServers": {
    "dgem": { "command": "/path/to/dgem/bin/dgem", "args": ["mcp", "--local"] }
  }
}
```

`--local` targets the engine on your machine only (no cloud credentials or fallback), so start the engine first.
opencode setup, other MCP clients, and troubleshooting:
[Studio, MCP and HTTP API](../reference/studio-mcp-api.md#3-model-context-protocol-mcp-server-dgem-mcp--post-mcp).

## Next

- Write your own policy: [Your first decision policy](../policies/first-policy.md).
- Need more speed or a shared endpoint? [Use a remote GPU](remote-gpu.md) or
  [deploy your own on Cloud Run](cloud-run.md).
