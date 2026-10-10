# dgem — DiffusionGemma as a Zero-Shot Decision Model

**`dgem`** turns **DiffusionGemma** (`26B-A4B-it`, a discrete-diffusion Gemma 4) into a **decision engine**. You describe a decision as a small template of typed questions (`boolean`, `choice`, `score`). `dgem` places every question on the model's bidirectional canvas and reads **a probability for every allowed answer, for every question, in one forward pass**. There is no free-text generation to parse, and every answer comes with a per-question uncertainty score you can use to decide when to act automatically and when to escalate.

📖 **Docs site:** [googlecloudplatform.github.io/dgem](https://googlecloudplatform.github.io/dgem/)

## What's in this repo

| Piece | What it does | Start here |
| :--- | :--- | :--- |
| **Decision engine** (`dgem decide`) | Compiles `.json.tmpl` *Policy-as-Template* files into one-pass multi-question readouts with per-option probabilities and Shannon entropy. | [The Journey to Decision Models](docs/decision-models-primer.md) · [Template Catalog](docs/policies/templates.md) |
| **Hesitation-gated decisions** | A probability for every option and a hesitation score from one pass; hesitant answers are handed off; calibration fitted per domain. | [Confidence beyond Shannon](docs/confidence/overview.md) · [Confidence and calibration](docs/confidence/index.md) |
| **Hesitation-gated cascade** | Answers low-hesitation decisions directly and hands the rest to Vertex AI `gemini-3.8-flash` with `dgem`'s probabilities attached. | [Authoring and cascades](docs/policies/authoring.md) |
| **Four surfaces** | CLI, HTTP gateway (`/api/decide`, `/v1/systemone`), MCP server (`dgem mcp`, `/mcp`), and the embedded Decision Studio web app (`dgem serve`). | [Studio, MCP & API](docs/reference/studio-mcp-api.md) · [CLI reference](docs/reference/cli.md) |
| **Evaluation tools** | `dgem bench-*` harnesses and a regression matrix (`scripts/bench_matrix.py`) to measure your own deployment and compare a candidate with a baseline. | [Evaluate a deployment](docs/deploy/evaluate.md) · [Regression matrix](docs/operate/regression-matrix.md) |
| **Serving** | Vertex AI Dedicated Endpoint on RTX PRO 6000, Cloud Run GPU (scale to zero), GCE VMs, and local Apple Silicon (Metal). | [From laptop to production](docs/deploy/index.md) |

## Quick Start

Full walkthrough: **[Run on your laptop](docs/deploy/laptop.md)**.

### 1. Start an Inference Backend
Choose the option matching your hardware:

* **Local Apple Silicon Mac (Metal)** — No Docker or cloud needed:
  ```bash
  make setup && make download && make local-up   # engine on :8080, Decision Studio on :8090
  ```
* **Any Linux workstation with an NVIDIA GPU (Docker)** — build the serving image into your own
  Artifact Registry first ([Build the serving image](docs/deploy/public-images.md)), then:
  ```bash
  docker run --gpus all -p 8080:8080 us-central1-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag>
  ```
* **Remote Google Cloud Endpoint**:
  ```bash
  export DGEM_VERTEX_URL="<ENDPOINT_ID>" GCP_PROJECT_NUMBER="<PROJECT_NUMBER>" DGEM_GCP_AUTH=1
  ```

### 2. Build CLI & Execute Your First Decision
```bash
make build                                   # compiles ./bin/dgem
./bin/dgem decide -t templates/support_triage.json.tmpl \
  -v 'ticket=I was billed $500 twice for my annual renewal this morning!' --stats
```

### 3. Open Decision Studio (Web UI)
```bash
./bin/dgem serve --port 8090
```
Open **[http://localhost:8090](http://localhost:8090)** to browse the templates, try Stage 2 Gemini cascades, and inspect OpenTelemetry trace waterfalls.

**Next, pick your path:** [build, deploy and operate](docs/deploy/index.md) ·
[confidence and calibration](docs/confidence/index.md) · [write your first policy](docs/policies/first-policy.md).

---

## Confidence beyond Shannon: hesitation-gated decisions

> **In one sentence:** every `dgem` answer comes with a probability for every allowed option and a **hesitation** score; clear answers are returned directly and hesitant ones are handed to a larger model or a person. Full explanation: **[`docs/confidence/overview.md`](docs/confidence/overview.md)**.

**Hesitation-gating.** Hesitation is Shannon entropy scaled to 0–100% by the number of options (bands: <16% clear, 16–50% somewhat unsure, >50% very unsure). Pick a threshold on a labelled sample of your own decisions, answer below it, and hand off above it.

**What keeps the probabilities trustworthy.**
1. **The answer template is part of the prompt.** Shared option letters across questions can make the model copy answers, and question ids that hint at an answer act as instructions. Use neutral, descriptive ids ([authoring](docs/policies/authoring.md)).
2. **Prompt layout.** By default the input comes before the questions (`document_first`); `schema_first` is available per request ([prompt layout](docs/policies/prompt-layout.md)).
3. **Order bias, measured.** The model can favour the first option when unsure; measure flips under shuffled option order on your policies.
4. **Calibration by domain.** The best temperature differs across domains, so fit one per policy ([calibrate your policy](docs/confidence/calibrate-your-policy.md)).
5. **Changes judged against measured noise.** Answers are not fully deterministic. The [regression matrix](docs/operate/regression-matrix.md) measures run-to-run agreement in every session and judges a candidate against your baseline.

---

## Four Ways to Use `dgem`

See **[Decision Studio, MCP and HTTP API](docs/reference/studio-mcp-api.md)** and **[Gateway and routing](docs/deploy/gateway.md)** for full details:

| Interaction Surface | Command / Endpoint | Description |
| :--- | :--- | :--- |
| **1. Decision Studio Web App** | `./bin/dgem serve --port 8090` | Embedded Lit web app with the `.json.tmpl` decision policies (`core`, `calibration`, `multimodal`, `rerank`), a backend target selector (`vertex_first` \| `vertex` \| `cloudrun`), Stage 2 Gemini cascades, approximate image box overlays (coarse, see [what dgem can do with images](docs/policies/images.md)), a plain-English **Concepts** tab, and OpenTelemetry trace waterfalls. |
| **2. Model Context Protocol (MCP)** | `./bin/dgem mcp` (`stdio`)<br>`POST /mcp` (Streamable HTTP) | MCP server with tools for policy and custom decisions, template listing, health and warm-up, and image localization, with `backend` selection and Stage 2 cascade options (`cascade_mode`, `cascade_threshold`, `cascade_model`). |
| **3. HTTP Gateway REST API** | `POST /api/decide/{template}`<br>`POST /v1/systemone`, `GET /api/templates` | Run any `.json.tmpl` policy or a `/v1/systemone` decision request with `X-DGem-Backend: vertex_first \| vertex \| cloudrun` (`X-DGem-Backend-Used` is returned on every response) and an optional Stage 2 `gemini-3.8-flash` cascade. |
| **4. CLI & evaluation harnesses** | `./bin/dgem decide --vertex-url ...`<br>`./bin/dgem bench-*` | Single-pass decisions (`--stats`, `--layout`, cascade flags) and evaluation harnesses (`bench`, `bench-intents`, `bench-calibration`, `bench-bbox`, `bench-bbox-judge`, `bench-guided`, `bench-vision`, `bench-jev`, `bench-decision-index`, `bench-permutation`). |

### Why a "Decision Model"?

Teams have historically chosen between two extremes for automated triage, routing, and guardrails:
1. **Discriminative classifiers and rule systems**: very fast, but **rigid**. A new policy rule or category means new labeled data, retraining, and redeployment.
2. **Autoregressive LLMs**: zero-shot flexible, but they generate answers token by token, can drift from the output format, and their token probabilities are spread across formatting tokens rather than the decision itself.

A **zero-shot decision model** sits in between. `dgem` compiles a `.json.tmpl` template into a fixed diffusion canvas with bidirectional attention; boolean gates, `[A–Z]` choices, and ordinal scores are read together in **one forward pass**, and every answer is constrained to the allowed options.

| Dimension | Discrete diffusion decision model (`dgem`) | Discriminative classifier | Autoregressive LLM |
| :--- | :--- | :--- | :--- |
| **Policy changes** | Edit the `.json.tmpl` template | New labels and retraining | Prompt engineering |
| **Passes per request** | One forward pass for all questions (cost grows with canvas length) | One classifier per attribute | One step per output token |
| **Joint conditioning** | Bidirectional across questions in a single pass | Independent heads | Left to right only |
| **Uncertainty** | Per-option probabilities and hesitation; per-domain temperature | Often overconfident out of distribution | Sequence log-probabilities over formatting tokens |

---

## Serving: From Laptop to Production

| Stage | Where | Guide |
| :--- | :--- | :--- |
| Crawl | Apple Silicon (`diffgemma`) or a local NVIDIA GPU | [Run on your laptop](docs/deploy/laptop.md) |
| Walk | Cloud Run GPU (1× RTX PRO 6000), scale to zero | [Deploy on Cloud Run](docs/deploy/cloud-run.md) |
| Run | Vertex AI dedicated endpoint, `g4-standard-48` + 1× RTX PRO 6000 | [Production on Vertex AI](docs/deploy/vertex.md) |

A gateway (`dgem serve`) in front routes to Vertex first and fails over to Cloud Run
([Gateway and routing](docs/deploy/gateway.md)). Capacity, runbook and observability:
[Latency and capacity](docs/operate/latency-capacity.md) · [Operations runbook](docs/operate/runbook.md).

---

## Installation & CLI Examples

```bash
# Clone the repository
git clone https://github.com/GoogleCloudPlatform/dgem.git
cd dgem

# Compile dgem binary into bin/
make build
```

### 1. Single-Pass Discrete Decision (`dgem decide`)
Evaluate customer tickets, code changes, or security alerts in a single forward pass:
```bash
./bin/dgem decide -t templates/support_triage.json.tmpl \
  -v 'ticket=I was billed $500 twice for my annual renewal this morning!' \
  --stats
```

Example output (illustrative):
```
QUESTION         | TYPE       | VALUE / CHOICE       | CONFIDENCE | ENTROPY (H) | AGREEMENT
-----------------------------------------------------------------------------------------
sentiment        | score      | frustrated           | 99.8%      | 0.002 nats  | 1.00
team             | choice     | billing              | 100.0%     | 0.000 nats  | 1.00
urgent           | boolean    | yes                  | 99.9%      | 0.001 nats  | 1.00
```

### 2. Generative Prompt Completion (`dgem ask`)
Standard chat completion with optional thinking mode:
```bash
./bin/dgem ask "Explain discrete block diffusion in two sentences."
```

### 3. Remote Endpoints with IAM Authentication
Connect to a remote GCE or Cloud Run GPU service:
```bash
./bin/dgem decide \
  -u "http://<EXTERNAL_IP>:8080/v1" \
  -m "nvidia/diffusiongemma-26B-A4B-it-NVFP4" \
  -t templates/support_triage.json.tmpl \
  -v 'ticket=Outage: production database cluster unreachable' \
  --stats
```

### 4. Multimodal Visual Assessment (`--image` / `-I`)
Attach local image paths (automatically base64 encoded) or remote URLs:
```bash
./bin/dgem decide -t templates/multimodal/ui_design_review.json.tmpl \
  -I fixtures/ui_component.png \
  -v 'component=CheckoutCard' \
  --stats
```

---

## Evaluating Your Deployment

The harnesses run against your own endpoint and write a JSON receipt you can replay later
(`--from-receipt`). Some datasets are fetched at run time instead of being committed
([Run a dataset](docs/policies/datasets.md)).

```bash
# Multi-field triage (support, code review, security; benchmarks/eval_dataset.jsonl)
./bin/dgem bench -u "<CLOUD_RUN_URL>/v1" --gcp-auth -d benchmarks/eval_dataset.jsonl -M slot -o bench.json

# Calibration and guardrail items (benchmarks/calibration_suite.jsonl, templates/calibration/)
./bin/dgem bench-calibration -u "<CLOUD_RUN_URL>/v1" --gcp-auth -w 4 -o calibration.json

# High-cardinality intents and out-of-scope routing (PolyAI/banking77, clinc150)
./bin/dgem bench-intents --dataset banking77 --full --workers 16

# JevBench public tasks (MIT; downloaded and verified by --sync)
./bin/dgem bench-jev --sync && ./bin/dgem bench-jev -u "<CLOUD_RUN_URL>/v1" --gcp-auth -o jev.json
```

To compare a new serving image or configuration with the one you run today, use the
[regression matrix](docs/operate/regression-matrix.md): it runs both in the same session and judges the
difference against the noise it measures. Dataset sources and licences: [`benchmarks/README.md`](benchmarks/README.md).

---

## Documentation

Published at [googlecloudplatform.github.io/dgem](https://googlecloudplatform.github.io/dgem/) ([index](docs/index.md)), organized by audience:

- **Build, deploy and operate:** [From laptop to production](docs/deploy/index.md) ·
  [Run on your laptop](docs/deploy/laptop.md) · [Use a remote GPU](docs/deploy/remote-gpu.md) ·
  [Build the serving image](docs/deploy/public-images.md) · [Deploy on Cloud Run](docs/deploy/cloud-run.md) ·
  [Production on Vertex AI](docs/deploy/vertex.md) · [Gateway and routing](docs/deploy/gateway.md) ·
  [Evaluate a deployment](docs/deploy/evaluate.md) · [Latency and capacity](docs/operate/latency-capacity.md) ·
  [Operations runbook](docs/operate/runbook.md) · [Observability](docs/operate/observability.md) ·
  [Regression matrix](docs/operate/regression-matrix.md)
- **Confidence and calibration:** [Overview](docs/confidence/index.md) ·
  [Calibrate your policy](docs/confidence/calibrate-your-policy.md) ·
  [Confidence beyond Shannon](docs/confidence/overview.md) ·
  [The journey to decision models](docs/decision-models-primer.md) · [Glossary](docs/glossary.md)
- **Policies and decisions:** [Your first decision policy](docs/policies/first-policy.md) ·
  [Authoring and Stage 2 cascades](docs/policies/authoring.md) · [Run a dataset](docs/policies/datasets.md) ·
  [Template catalog](docs/policies/templates.md) · [Real-world applications](docs/policies/applications.md) ·
  [Taxonomy discovery](docs/policies/taxonomy-discovery.md) · [Images](docs/policies/images.md)
- **Reference:** [CLI, HTTP and MCP](docs/reference/cli.md) · [Studio, MCP and HTTP API](docs/reference/studio-mcp-api.md) ·
  [Vertex AI vs. Cloud Run](docs/reference/vertex-vs-cloud-run.md) ·
  [Apple Silicon engine](docs/reference/metal-engine.md) · [How the model decides in one pass](docs/confidence/architecture.md)

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contributor license agreement (CLA) requirements and community guidelines, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for our Code of Conduct. If you encounter a bug or have feedback on templates or evaluation tools, please open an [Issue](https://github.com/GoogleCloudPlatform/dgem/issues).

## License

This project is licensed under the [Apache-2.0 License](LICENSE). Evaluation datasets derived from third-party sources are located in [`third_party/`](third_party/) under their respective licenses; see [`benchmarks/README.md`](benchmarks/README.md).

## Disclaimer

> [!CAUTION]
> This is **not** an officially supported Google product.
> This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
