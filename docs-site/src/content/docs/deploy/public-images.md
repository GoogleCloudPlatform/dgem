---
title: "Build the Serving Image"
description: "How to build the dgem DiffusionGemma serving image (lean or with baked weights) into your own Artifact Registry with Cloud Build, run it on a GPU host, and configure it."
---

`dgem` does not publish prebuilt container images yet; prebuilt images may be published later. Until then you build
the serving image from this repository into an Artifact Registry repository in your own Google Cloud project. The
build runs on Cloud Build, so you don't need Docker or a GPU on your workstation.

There are two images, both defined in `deploy/cloudrun/`:

| Image | Dockerfile | Weights | Use it for |
| :--- | :--- | :--- | :--- |
| **Lean** (`dgemma`) | `deploy/cloudrun/Dockerfile` | Not in the image. At boot the container uses weights mounted at `MODEL`, stages them from a `gs://` URI (`DGEM_WEIGHTS_URI`, or `AIP_STORAGE_URI` on Vertex AI), or downloads the public NVFP4 checkpoint from Hugging Face (`MODEL_HF`) | Cloud Run GPU (recommended), Vertex AI, GCE VMs, local Docker |
| **Baked weights** | `deploy/cloudrun/Dockerfile.weights` | NVFP4 weights copied into the image at `/opt/dgemma/weights` at build time | Hosts without access to Cloud Storage or Hugging Face at boot (offline or air-gapped). Much larger, so pulls are slow |

The lean image is built on the official `vllm/vllm-openai` nightly image, pinned by commit and digest, and adds
`structured_server.py`, `entrypoint.sh` and a static `dgem` binary. A running container reports its build in
`/health` (`version`, `revision`, `vllm_commit`).

---

## 1. Prerequisites

* A Google Cloud project with billing, and the Cloud Build and Artifact Registry APIs enabled.
* `gcloud` authenticated against that project, and Go (the build script compiles a static Linux `dgem` binary
  into the build context first).
* Run the commands from the repository root.

## 2. Build the lean image

```bash
GCP_PROJECT=<PROJECT> make cloudrun-build
```

`make cloudrun-build` runs `scripts/build_cloudrun_image.sh`, which:

1. creates the Artifact Registry Docker repository if it does not exist;
2. builds a static `dgem` binary into `deploy/cloudrun/dgem` (removed again on exit);
3. submits `deploy/cloudrun/` to Cloud Build with `deploy/cloudrun/cloudbuild.yaml`, which bakes the version
   (`git describe`) and revision into the image labels and environment;
4. prints the image URI and the matching `make cloudrun-deploy` command.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GCP_PROJECT` | `gcloud config get-value project` | Project that hosts the build and the registry. |
| `GCP_REGION` | `us-central1` | Artifact Registry location. |
| `AR_REPOSITORY` | `dgem` | Artifact Registry repository name. |
| `IMAGE_NAME` | `dgemma` | Image name. |
| `IMAGE_TAG` | `git rev-parse --short HEAD` | Image tag. |

The result is `<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<commit>`. The deploy scripts default to the `:latest`
tag of that image (`us-central1-docker.pkg.dev/<PROJECT>/dgem/dgemma:latest`), so either build with
`IMAGE_TAG=latest` or, better, pass the commit tag explicitly when you deploy (`CLOUDRUN_IMAGE=...` for Cloud Run,
`IMAGE_URI=...` for Vertex AI). Pin a tag or digest in production.

You can also submit the Cloud Build config directly (build the static binary into `deploy/cloudrun/dgem` first,
as the script does):

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o deploy/cloudrun/dgem .
gcloud builds submit deploy/cloudrun --project=<PROJECT> \
  --config=deploy/cloudrun/cloudbuild.yaml \
  --substitutions=_IMAGE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma,_TAG=$(git rev-parse --short HEAD),_VERSION=$(git describe --tags --always)
rm deploy/cloudrun/dgem
```

## 3. Stage the weights in Cloud Storage

The lean image needs the weights at boot. For Cloud Run and Vertex AI, stage them once in a bucket in your project:

```bash
GCP_PROJECT=<PROJECT> make cloudrun-stage
```

`make cloudrun-stage` runs `scripts/stage_model_gcs.sh`, which creates `gs://dgem-weights-<PROJECT>` (override
with `GCS_BUCKET`, region with `GCP_REGION`) and copies `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (override with
`MODEL_ID`) to `gs://dgem-weights-<PROJECT>/dgemma/` using Cloud Build, so nothing is downloaded to your
workstation. It uses `HF_TOKEN` from the environment or `.env` if set, and exits early if the weights are
already there. Never commit `.env`.

## 4. Deploy

```bash
GCP_PROJECT=<PROJECT> CLOUDRUN_IMAGE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag> make cloudrun-deploy
```

`make cloudrun-deploy` runs `scripts/deploy_cloudrun_vllm.sh`; see [Deploy on Cloud Run](/dgem/deploy/cloud-run/) for its
options. For Vertex AI pass the same image as `IMAGE_URI` to `scripts/deploy_vertex_endpoint.sh`; see
[Production on Vertex AI](/dgem/deploy/vertex/).

## 5. Build the baked-weights image (optional)

`deploy/cloudrun/Dockerfile.weights` starts from a lean image (`BASE` build argument) and downloads the NVFP4
checkpoint into `/opt/dgemma/weights` at build time, setting `MODEL=/opt/dgemma/weights`. Build the lean image
first (section 2), then build on top of it with `deploy/cloudrun/cloudbuild-weights.yaml`. It reads
`<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag>` and pushes `.../dgem/dgemma-weights:<tag>` in the project
you submit to (override `_REGISTRY`, `_BASE_IMAGE` or `_IMAGE` for other names):

```bash
gcloud builds submit deploy/cloudrun --project=<PROJECT> \
  --config=deploy/cloudrun/cloudbuild-weights.yaml \
  --substitutions=_TAG=<tag of the lean image>
```

On Cloud Run, prefer the lean image with weights staged in Cloud Storage: the baked image is large, so a fresh
instance spends minutes pulling it and cold starts are not faster.

---

## 6. Running on your own GPU host

**GPU requirements.** The image serves the 4-bit NVFP4 checkpoint (~18 GiB of weights):

| GPU | Status | Host RAM |
| :--- | :--- | :--- |
| **NVIDIA RTX PRO 6000 (Blackwell, 48 GB)** | **Recommended; tested** (native FP4, vision tower on) | 80 GiB |
| NVIDIA L4 (Ada, 24 GB) | Works, slower per request | 32 GiB with `DISABLE_MM=1`; 64 GB with the vision tower on |
| Other NVIDIA GPUs (for example H100, A100, other Blackwell cards) | Not tested | — |

On an untested GPU, verify the install before relying on it: [Evaluate dgem on your own GPU](/dgem/deploy/evaluate/).

Pull the image you built (`gcloud auth configure-docker <REGION>-docker.pkg.dev` once), then run it.

### A. Standard mode (`ROLE=default`)

Serves `/v1/chat/completions` (OpenAI format), `/v1/systemone`, and `/health` on port `8080`:

```bash
docker run --gpus all -p 8080:8080 \
  <REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag>
```

With no weights mounted or staged, the container downloads them from Hugging Face at boot. To use a local copy,
mount it and set `MODEL`, for example `-v /data/dgemma:/weights -e MODEL=/weights`. On a Google Cloud VM you can
instead set `-e DGEM_WEIGHTS_URI=gs://dgem-weights-<PROJECT>/dgemma`; staging from Cloud Storage authenticates
through the VM's metadata server, so it only works on Google Cloud.

Then point the `dgem` CLI, Decision Studio, or MCP at `http://127.0.0.1:8080/v1`:

```bash
# Single decision via CLI
./bin/dgem decide -u http://127.0.0.1:8080/v1 \
  -t templates/support_triage.json.tmpl \
  -v 'ticket=Emergency database outage' --stats

# JevBench (public MIT benchmark) run against your server
./bin/dgem bench-jev -u http://127.0.0.1:8080/v1 --endpoint-kind gpu
```

### B. Decision-protocol adapter mode (`ROLE=decision-index`)

Starts `dgem systemone serve` on port `8080` (multi-slot canvas batching and two-stage bracket routing for large
option sets) in front of the internal `structured_server.py` on port `8081` and vLLM on port `8000`. Clients that
speak the `/v1/systemone` decision protocol send requests to `http://127.0.0.1:8080`:

```bash
docker run --gpus all -p 8080:8080 \
  -e ROLE=decision-index \
  -e TEMPERATURE=1.0 \
  <REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag>
```

---

## 7. Environment variables and runtime configuration

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| **`ROLE`** | `default` | `decision-index` (starts the `dgem systemone serve` adapter on `PORT`) or `default` (direct serving). |
| **`PORT`** | `8080` | External HTTP port to bind. |
| **`TEMPERATURE`** | `1.0` | Slot logit temperature used by the `decision-index` adapter ($1.0$ = unscaled). |
| **`API_KEY`** | `""` (open) | Optional secret key. When set, all incoming requests must supply `Authorization: Bearer <key>`. |
| **`DGEM_SERVER_KEY`** | `""` (off) | Service-to-service secret. When set, every `POST` needs header `X-DGem-Key` with one of the keys (comma-separated to rotate; callers send the first entry); `GET /health` stays open. Use it when the serving image is reachable by more than your own services, for example a Vertex dedicated endpoint: give it to services that call the server (your gateway, batch jobs, probes), not to people, and route people through an authenticated gateway so access is tracked. Callers send it when their own `DGEM_SERVER_KEY` is set (`dgem` CLI and gateway, `dgem systemone serve`, `scripts/matrix`, `scripts/probe.py`, `scripts/serving_speed.py`, `scripts/contract_diff.py`). Unlike `API_KEY` it works on Vertex and Cloud Run, where `Authorization` carries the Google token. The deploy scripts set it from the Secret Manager secret `dgem-server-key` when it exists ([Vertex](/dgem/deploy/vertex/#1-prerequisites)). |
| **`DGEM_IMAGE_URLS`** | `deny` | Whether the server lets vLLM fetch http(s) image URLs from requests. `deny` accepts images only as data: URIs or multipart uploads and answers 400 to a URL: the serving image has network egress, so a URL could be steered to the metadata server or internal hosts. The `dgem` CLI and gateway fetch URLs themselves and send data: URIs, so they work with `deny`. `allow` restores the old behaviour for trusted callers only. |
| **`CANVAS`** | `128` | Served diffusion canvas length in tokens. |
| **`MODEL`** | `/mnt/gcs/dgemma` | Weights path. Baked weights at `/opt/dgemma/weights` are used automatically; a mounted path is used if it has `config.json`; otherwise weights are staged from `DGEM_WEIGHTS_URI` (a `gs://` URI) or downloaded from `MODEL_HF`. |
| **`MODEL_HF`** | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | Hugging Face repo downloaded when no weights are baked, mounted or staged. |
| **`DEFAULT_SAMPLES`**, **`MAX_INFLIGHT`** | `1`, `8` | Samples for schemas without `samples`; decisions processed at once (the rest queue). |
| **`MAX_MODEL_LEN`** | `4096` (image); deploy scripts set `8192` on RTX PRO 6000 | Longest prompt plus answer, in tokens. Longer prompts are refused. |
| **`DEFAULT_ISOLATE`**, **`ISOLATE_AUTO_MAX`**, **`ISOLATE_MAX_QUESTIONS`** | `auto`, `3`, `8` | Which questions get a read of their own when a request doesn't say: `auto` (yes/no questions in text requests of 2–3 questions with at least 2 yes/no; from v0.3.0), `none` (one joint read, the behaviour before v0.3.0), `noul` (every yes/no question, requests of 2–8 questions) or `all`. A request can set `"isolate"`, and a question `"alone": true/false`. See [reading yes/no questions alone](/dgem/policies/prompt-layout/#yesno-questions-in-small-requests-are-read-alone). |
| **`HTTP_IDLE_TIMEOUT`** | `120` | Seconds before an idle keep-alive connection is closed (0 = never). |
| **`DEFAULT_LABELS`** | `az` | Labels for `choice` options in requests without `labels`: `az` (A–Z, at most 26 options), `az_aa` (A–Z then AA–AZ, at most 52) or `az_lower` (A–Z then a–z, at most 52, requests of up to 10 questions only). Experimental beyond `az`; keep the default unless you have measured otherwise. |
| **`DEFAULT_SEED_TOKEN`** | `random` | What sits at each answer position before the read, for requests without `seed_token`: `random` (a seeded random token, so reads vary with the seed) or `fixed` (the padding token; reads stop depending on the seed). Experimental; keep `random` unless you have measured otherwise. |
| **`DEFAULT_LAYOUT`** | `document_first` | Prompt layout for requests without `layout`: `document_first` (the state, then the questions) or `schema_first` (the questions as the system prompt, the behaviour before v0.2.0). |
| **`DISABLE_MM`** | `0` | `1` disables the vision tower (required on an L4 with 32 GiB host RAM). |

---

## 8. Container lineage

* **Base image**: official `vllm/vllm-openai:nightly`, pinned by commit and digest (`vllm-project/vllm:main` commit
  `a9eafde59cb`), which includes [DiffusionGemma](https://gemma.ai/models/diffusiongemma/) structured generation and constrained reads. Structured mode lets the server send
  its own seeded canvas and ask for a read-only denoise step with logprobs for chosen token ids (`logprob_token_ids`);
  constrained reads compute the output layer and sampler over those label ids only, which gives the same answer with
  about a quarter less GPU time per read. vLLM accepts at most 128 such ids per request.
* **Patches**: `structured_server.py` is vLLM's upstream example plus `deploy/cloudrun/server/dgem.patch`; the
  upstream revision is tracked in `deploy/cloudrun/server/UPSTREAM`.
* **Release notes**: [CHANGELOG](https://github.com/GoogleCloudPlatform/dgem/blob/main/CHANGELOG.md).
