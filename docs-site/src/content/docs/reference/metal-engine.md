---
title: Setup & Metal Engine Guide
description: Step-by-step instructions for running DiffusionGemma locally on Apple Silicon Metal.
---

This guide covers setting up, running, and querying **DiffusionGemma 26B-A4B** locally on Apple Silicon (tested on Apple M5 with 32 GB unified memory) using **`diffgemma`** (a native Rust + Metal inference engine) and **`dgem`** (a Go-based CLI assistant for structured decisions and generative queries).

---

## 1. Overview & Architectural Concepts

**DiffusionGemma** (developed by Google DeepMind) is a 26B-parameter Mixture-of-Experts (MoE) foundation model with 3.8B active parameters (8 active experts out of 128 total + 1 shared expert).

Unlike traditional autoregressive language models that generate text token-by-token from left to right, DiffusionGemma utilizes **discrete block diffusion**:
* **Canvas-based generation**: It operates on a 256-token canvas with bidirectional cross-attention.
* **Parallel denoising**: It iteratively denoises blocks of tokens in parallel, generating 15–20 tokens per forward pass.
* **Fast structured decisions ("System-1" Judgment)**: By seeding a canvas with a predefined JSON template and leaving answer slots as noise, the model can evaluate classification, categorization, and scale choices in **a single forward pass (~880 ms)** without generating conversational filler.

### Moving Beyond "Noul": The Boolean Decision

In early structured-decision schemas, binary decisions were labeled with the neologism `"noul"`. In practical computing and schema design, this is simply a **boolean** (or binary predicate decision):
* You provide a question or proposition (e.g. *"Does this ticket require immediate escalation?"*).
* The model evaluates whether the proposition holds (`yes` vs. `no`, `true` vs. `false`), providing calibrated confidence, entropy, and error bars.
* The engine natively accepts `"boolean"` and `"bool"` interchangeably with `"noul"`.

---

## 2. Hardware & Memory Tuning for 32 GB Macs

DiffusionGemma's full-precision BF16 checkpoint is ~50 GB, which cannot fit into 32 GB of RAM. However, the 4-bit quantized pack (`mmastrac/diffgemma-26b-a4b-it-q4`) compresses the MoE experts into 4-bit affine blocks while keeping precision-sensitive attention, norms, and embeddings in 16-bit, resulting in an **18.84 GiB** resident footprint.

### Unified Memory Budgeting

On macOS Apple Silicon, unified memory is shared between the CPU and the Metal GPU. To prevent macOS memory pressure or swap file thrashing:

| Configuration | Model Weights | KV Cache (Working Set) | Total Memory | Safe for 32 GB Mac? |
| :--- | :--- | :--- | :--- | :--- |
| `--ctx 131072` (default 128k) | 18.84 GiB | ~12–14 GiB | ~31–33 GiB | ⚠️ Near boundary, may swap |
| **`--ctx 32768` (Recommended)** | **18.84 GiB** | **~2.5 GiB** | **~21.3 GiB** | **✅ Optimal (leaves ~10 GB for OS)** |
| `--ctx 16384` (Lightweight) | 18.84 GiB | ~1.2 GiB | ~20.0 GiB | **✅ Very safe** |

---

## 3. Prerequisites

Ensure you have the following installed on your Mac:
* **macOS 15+** with Apple Silicon (M1/M2/M3/M4/M5).
* **Xcode Command Line Tools**: `xcode-select --install`
* **Rust toolchain (1.85+)**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
* **Go (1.22+)**: `brew install go`
* **cURL** and **jq**: standard on macOS.

---

## 4. Installation & Model Download

### Step 1: Install `diffgemma`
Compile and install the Metal-accelerated inference binary directly from upstream (or run `make setup`):

```bash
cargo install --git https://github.com/mmastrac/diffgemma diffgemma
```

Verify the binary is in your `$PATH`:
```bash
diffgemma
```

### Step 2: Download Model Weights
Run the built-in downloader to fetch the canonical 4-bit quantization pack from Hugging Face (`mmastrac/diffgemma-26b-a4b-it-q4`). Use `--jobs 8` to enable parallel chunk downloads (or run `make download`):

```bash
# In your project directory (or 'make download'):
diffgemma download --jobs 8
```

This downloads 76 chunks totaling **18.84 GiB** into `model/diffgemma-26b-a4b-it-q4/` and verifies the blob against internal SHA256 checksums.

### Step 3: Run a Sanity Check
Test generation directly from the command line:

```bash
diffgemma ask -m model/diffgemma-26b-a4b-it-q4 --ctx 16384 --max-new-tokens 128 \
  -p "Explain discrete text diffusion in two sentences."
```

*Note: On the first run, `diffgemma` compiles the Metal shaders for your specific GPU core configuration and saves the binary archive to `~/.cache/diffgemma/metal-pipelines/`. Subsequent runs load immediately in fractions of a second.*

---

## 5. Running the Local OpenAI-Compatible Server

`diffgemma serve` exposes an OpenAI-compatible HTTP server (`POST /v1/chat/completions`) capable of handling both standard conversational chat and structured decisions.

### Starting and Stopping Local Services

You can manage the standalone `diffgemma` engine or the full local stack (engine + Decision Studio Gateway) using `Makefile` targets:

```bash
# Full local stack: diffgemma engine (:8080) + dgem gateway & Web Studio (:8090)
make local-up      # Starts both services in background
make local-status  # Displays health and PID status of both services
make local-down    # Cleanly stops both services

# Standalone diffgemma engine (:8080 only)
make serve         # Starts diffgemma serve in background (writes diffgemma.pid and server.log)
make stop          # Stops the background diffgemma engine process

# Standalone dgem gateway (:8090 only, targets existing diffgemma)
make gateway-up    # Starts dgem serve gateway (:8090)
make gateway-down  # Stops the dgem gateway process
```

Or run directly:
```bash
diffgemma serve \
  -m model/diffgemma-26b-a4b-it-q4 \
  --ctx 32768 \
  --addr 127.0.0.1:8080 > server.log 2>&1 &
```

### Verifying Server Health
```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
```
Expected response:
```json
{
  "data": [
    {"id": "diffgemma-26b-a4b-it-q4", "object": "model", "owned_by": "local"},
    {"id": "diffgemma-26b-a4b-it-q4:think", "object": "model", "owned_by": "local"},
    {"id": "diffgemma-26b-a4b-it-q4:think=false", "object": "model", "owned_by": "local"}
  ],
  "object": "list"
}
```

> **Cloud Deployment**: If you want to host DiffusionGemma on Google Cloud Run with an NVIDIA RTX Pro 6000 or L4 GPU instead of running locally, see the [Remote Endpoints & Cloud Deployment Guide](/dgem/deploy/remote-gpu/) or run `make cloudrun-deploy`.

---

## 6. Discrete Diffusion Slot Readout (Single-Pass Decisions)

A request containing a JSON question schema in the `system` role automatically triggers the discrete diffusion slot-readout pathway. The user message provides the target state or text.

### Supported Question Types

1. **`boolean` (or `bool` / `noul`)**:
   Binary proposition (`yes` / `no`). Returns `probabilities.yes`, `probabilities.no`, and confidence.
2. **`choice`**:
   Categorical selection from a list of named options (up to 26 single-token labels `A`, `B`, `C`...).
3. **`score`**:
   Ordered scalar evaluation (e.g. `["calm", "frustrated", "furious"]`). Returns the expected numerical level and the top level.

### Example cURL Request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "{\"instructions\": \"Triage customer tickets\", \"questions\": [{\"id\": \"urgent\", \"type\": \"boolean\", \"instructions\": \"Immediate escalation required?\"}, {\"id\": \"team\", \"type\": \"choice\", \"instructions\": \"Owning team\", \"options\": [{\"name\": \"billing\"}, {\"name\": \"support\"}, {\"name\": \"engineering\"}]}, {\"id\": \"sentiment\", \"type\": \"score\", \"instructions\": \"Customer distress\", \"levels\": [\"calm\", \"frustrated\", \"furious\"]}]}"
      },
      {
        "role": "user",
        "content": "{\"ticket\": \"Our production database is returning 500 across all cluster nodes!\"}"
      }
    ]
  }' | jq .
```

### Understanding the Response Diagnostics
The response includes `answers` and a comprehensive `diagnostics` object:
* **`confidence`**: Mean probability of the winning choice.
* **`stderr`**: Standard error computed across noise draws.
* **`agreement`**: Proportion of noise draws that agreed on the winning label.
* **`timing.reused_tokens`**: Indicates that the system prompt schema was reused directly from the KV cache (zero-recomputation prefill).
* **`samples.policy`**: When set to `"auto"`, the engine stops at 1 read if entropy is low (<0.10 nats), or automatically gathers up to 4 reads if ambiguity is detected.

---

## 7. Connecting to Opencode

To use your local `diffgemma` model directly within `opencode`:

```bash
OPENCODE_CONFIG_CONTENT='{
  "provider": {
    "diffgemma": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "diffgemma (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "unused" },
      "models": { "diffgemma-26b-a4b-it-q4": { "name": "DiffGemma 26B-A4B q4" } }
    }
  }
}' opencode -m diffgemma/diffgemma-26b-a4b-it-q4
```

---

## 8. The `dgem` CLI Tool

`dgem` is the dedicated Go CLI for managing templates, running structured reads, and inspecting execution telemetry.

### Build and Install
```bash
make build
```

### Basic Usage
```bash
# Run a structured decision with detailed stats
./bin/dgem decide -t templates/support_triage.json.tmpl -v ticket="System is completely down" --stats

# Run a generative prompt
./bin/dgem ask "Explain the difference between autoregression and diffusion." --stats

# Render a template without executing
./bin/dgem template render -t templates/code_review.json.tmpl -v diff="added auth middleware"

# Run the comparative benchmark
./bin/dgem bench
# Or: make bench

# Deploy to Google Cloud Run with GPU
make cloudrun-deploy
```
