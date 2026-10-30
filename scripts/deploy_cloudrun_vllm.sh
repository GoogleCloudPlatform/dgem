#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Load .env if present
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
elif [[ -f "../.env" ]]; then
  set -a
  source "../.env"
  set +a
fi

# Configuration
PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${CLOUDRUN_SERVICE_NAME:-dgemma}"
GPU_TYPE="${CLOUDRUN_GPU_TYPE:-nvidia-rtx-pro-6000}" # nvidia-rtx-pro-6000 (tested, vision on) or nvidia-l4 (set DISABLE_MM=1)
BUCKET="${GCS_BUCKET:-dgem-weights-${PROJECT_ID}}"

# Image: CLOUDRUN_IMAGE, or the image built by `make cloudrun-build` in this project.
DEFAULT_AR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/dgem/dgemma:latest"
IMAGE_TAG="${CLOUDRUN_IMAGE:-$DEFAULT_AR_IMAGE}"
if [[ -n "$PROJECT_ID" && -z "${CLOUDRUN_IMAGE:-}" ]] && ! gcloud artifacts docker images describe "$DEFAULT_AR_IMAGE" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "Error: $DEFAULT_AR_IMAGE not found. Build it first (make cloudrun-build) or set CLOUDRUN_IMAGE." >&2
  exit 1
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No Google Cloud Project ID detected."
  echo "Set GCP_PROJECT=<project-id> or run 'gcloud config set project <project-id>'."
  exit 1
fi

# The IAM step at the end (setup_cloudrun_iam.sh) grants invoker access to this group; check it up front
# instead of failing after the deploy.
if [[ -z "${ALLOW_GROUP:-}" ]]; then
  echo "Error: ALLOW_GROUP is required (the Google group or user allowed to call the service, e.g. my-team@example.com)." >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud CLI not found. Please install the Google Cloud SDK."
  exit 1
fi

echo "================================================================================"
echo "  Deploying DiffusionGemma (dgemma) to Google Cloud Run with GPU"
echo "================================================================================"
echo "Project:      $PROJECT_ID"
echo "Region:       $REGION"
echo "Service Name: $SERVICE_NAME"
echo "GPU Type:     $GPU_TYPE"
echo "Image:        $IMAGE_TAG"
echo "Weights GCS:  gs://${BUCKET}/dgemma/"
echo "================================================================================"

# 1. Ensure GCS bucket and weights exist
if ! gcloud storage ls "gs://${BUCKET}/dgemma/config.json" >/dev/null 2>&1; then
  echo "==> Weights not found at gs://${BUCKET}/dgemma/."
  echo "==> Automatically staging weights via scripts/stage_model_gcs.sh..."
  ./scripts/stage_model_gcs.sh "$BUCKET" "$REGION"
fi

# 2. Ensure least-privilege GPU Service Account (dgemma-gpu-sa) exists and has read-only access to weights bucket
GPU_SA_NAME="${GPU_SA_NAME:-dgemma-gpu-sa}"
GPU_SA="${GPU_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${GPU_SA}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "==> Creating least-privilege Service Account ${GPU_SA}..."
  gcloud iam service-accounts create "${GPU_SA_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="DiffusionGemma GPU Engine SA (Read-Only GCS Weights)"
fi
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${GPU_SA}" \
  --role="roles/storage.objectViewer" \
  --quiet >/dev/null

# 3. Adaptive CPU, Memory, and SHM Staging based on GPU tier
if [[ "$GPU_TYPE" == "nvidia-rtx-pro-6000" ]]; then
  CPU="20"
  MEMORY="80Gi"
  COPY_SHM="${COPY_TO_SHM:-1}" # Stage sequentially into /tmp/dgemma (80Gi tmpfs) in ~55s instead of 429s FUSE random mmap
  CANVAS_LEN="128"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" # same short-prompt accuracy/latency as 4096, longer prompts answered
else
  # Default: 1x NVIDIA L4 (24GB VRAM, 32GB host RAM)
  CPU="8"
  MEMORY="32Gi"
  COPY_SHM="0" # Direct GCS FUSE buffered streaming without filling 32GB RAM
  CANVAS_LEN="32"
  MAX_MODEL_LEN="32768"
fi

DISABLE_MM_VAL="${DISABLE_MM:-0}"
ENV_VARS="MODEL=/mnt/gcs/dgemma,CANVAS=${CANVAS_LEN},MAX_SEQS=32,MAX_MODEL_LEN=${MAX_MODEL_LEN},GPU_UTIL=0.40,KV_CACHE_GB=2,ATTN=TRITON_ATTN,ENFORCE_EAGER=1,DISABLE_MM=${DISABLE_MM_VAL},TORCH_COMPILE_DISABLE=1,VLLM_WORKER_MULTIPROC_METHOD=fork,CUDA_MODULE_LOADING=LAZY,COPY_TO_SHM=${COPY_SHM}"
ENV_VARS="${ENV_VARS},DEFAULT_SAMPLES=${DEFAULT_SAMPLES:-1},MAX_INFLIGHT=${MAX_INFLIGHT:-8}"
# HF_TOKEN comes from Secret Manager (never a plaintext env var). Create once with:
#   printf %s "$HF_TOKEN" | gcloud secrets create dgemma-hf-token --data-file=- && \
#   gcloud secrets add-iam-policy-binding dgemma-hf-token --member=serviceAccount:$GPU_SA --role=roles/secretmanager.secretAccessor
HF_SECRET="${HF_SECRET:-dgemma-hf-token}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENTRYPOINT_B64="$(base64 < "${REPO_ROOT}/deploy/cloudrun/entrypoint.sh" | tr -d '\n')"
BOOT_CMD="echo '${ENTRYPOINT_B64}' | base64 -d > /tmp/run.sh && chmod +x /tmp/run.sh && exec /tmp/run.sh"

DEPLOY_FLAGS=(
  "--project" "$PROJECT_ID"
  "--region" "$REGION"
  "--image" "$IMAGE_TAG"
  "--service-account" "$GPU_SA"
  "--execution-environment" "gen2"
  "--no-allow-unauthenticated"
  "--command=/bin/bash"
  "--args=^@^-c@${BOOT_CMD}"
  "--cpu" "$CPU"
  "--memory" "$MEMORY"
  "--gpu" "1"
  "--gpu-type" "$GPU_TYPE"
  "--no-gpu-zonal-redundancy"
  "--no-cpu-throttling"
  "--min-instances" "${CLOUDRUN_MIN_INSTANCES:-0}"
  "--max-instances" "1"
  "--concurrency" "32"
  "--port" "8080"
  "--add-volume=name=weights,type=cloud-storage,bucket=${BUCKET},readonly=true,mount-options=enable-buffered-read=true"
  "--add-volume-mount=volume=weights,mount-path=/mnt/gcs"
  "--startup-probe=httpGet.path=/health,httpGet.port=8080,initialDelaySeconds=10,periodSeconds=5,timeoutSeconds=4,failureThreshold=120"
  "--set-env-vars=${ENV_VARS}"
)
# Direct VPC egress with Private Google Access: staging 17.53 GiB of weights from Cloud Storage drops from
# ~400 s (public egress, ~46 MiB/s) to ~83 s (~410 MiB/s), cutting cold start from ~7.5 to ~2.5 minutes.
# Requires Private Google Access on the subnet. With all-traffic egress and no Cloud NAT the service has no
# public internet access (it only needs Cloud Storage and the metadata server). Set CLOUDRUN_VPC_EGRESS=off to disable.
CLOUDRUN_NETWORK="${CLOUDRUN_NETWORK:-default}"
CLOUDRUN_SUBNET="${CLOUDRUN_SUBNET:-default}"
CLOUDRUN_VPC_EGRESS="${CLOUDRUN_VPC_EGRESS:-all-traffic}"
if [[ "$CLOUDRUN_VPC_EGRESS" != "off" ]]; then
  DEPLOY_FLAGS+=("--network=${CLOUDRUN_NETWORK}" "--subnet=${CLOUDRUN_SUBNET}" "--vpc-egress=${CLOUDRUN_VPC_EGRESS}")
fi
SECRETS=()
if gcloud secrets describe "$HF_SECRET" --project "$PROJECT_ID" >/dev/null 2>&1; then
  SECRETS+=("HF_TOKEN=${HF_SECRET}:latest")
fi
# Service-to-service key for the serving image (X-DGem-Key; see docs/deploy/public-images.md). Held only by services
# (gateway, matrix jobs, probes), never given to people. Skipped when the secret doesn't exist.
SERVER_KEY_SECRET="${SERVER_KEY_SECRET:-dgem-server-key}"
if gcloud secrets describe "${SERVER_KEY_SECRET}" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud secrets add-iam-policy-binding "${SERVER_KEY_SECRET}" --project "$PROJECT_ID" \
    --member="serviceAccount:${GPU_SA}" --role=roles/secretmanager.secretAccessor --quiet >/dev/null
  SECRETS+=("DGEM_SERVER_KEY=${SERVER_KEY_SECRET}:latest")
else
  echo "==> WARNING: secret ${SERVER_KEY_SECRET} not found: the service will accept POSTs without X-DGem-Key"
fi
if [ ${#SECRETS[@]} -gt 0 ]; then
  DEPLOY_FLAGS+=("--set-secrets=$(IFS=,; echo "${SECRETS[*]}")")
fi

# CLOUDRUN_TAG=<tag>: deploy as a tagged revision with no traffic (blue/green); verify at
# https://<tag>---<service-host>, then: gcloud run services update-traffic <service> --to-tags=<tag>=100
# Always update the service through this script: it embeds the repository's entrypoint.sh in the revision's
# args, and `gcloud run services update --image` would keep the previous revision's (older) entrypoint.
if [[ -n "${CLOUDRUN_TAG:-}" ]]; then
  DEPLOY_FLAGS+=("--no-traffic" "--tag=${CLOUDRUN_TAG}")
fi

echo "==> Deploying to Cloud Run..."
gcloud beta run deploy "$SERVICE_NAME" "${DEPLOY_FLAGS[@]}"

# Bind invoker permissions (dgemma-gateway-sa and ALLOW_GROUP)
if [[ -x "./scripts/setup_cloudrun_iam.sh" ]]; then
  GCP_PROJECT="$PROJECT_ID" GCP_REGION="$REGION" UPSTREAM_SERVICE="$SERVICE_NAME" ./scripts/setup_cloudrun_iam.sh
fi

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format="value(status.url)")

echo ""
echo "================================================================================"
echo "  Deployment Successful! Zero-to-N Autoscaling Active"
echo "================================================================================"
echo "Service URL:     $SERVICE_URL"
echo "Service Account: $GPU_SA (Read-only on gs://${BUCKET}, IAP OFF)"
echo ""
echo "Query using dgem with automatic IAM identity authentication:"
echo "  ./bin/dgem decide -u \"${SERVICE_URL}/v1\" --gcp-auth \\"
echo "    -t templates/support_triage.json.tmpl \\"
echo "    -v 'ticket=Emergency: cluster outage' --stats"
echo ""
echo "Deploy or update the IAP-ready Web Decision Studio Gateway (dgemma-gateway):"
echo "  make gateway-deploy"
echo "================================================================================"
