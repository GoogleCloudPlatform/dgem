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

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$PROJECT" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi
REGION="${GCP_REGION:-us-central1}"
GATEWAY_SERVICE="${GATEWAY_SERVICE:-dgemma-gateway}"
UPSTREAM_SERVICE="${UPSTREAM_SERVICE:-dgemma}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
IMAGE="us-central1-docker.pkg.dev/${PROJECT}/dgem/dgemma-gateway:${IMAGE_TAG}"
ALLOW_GROUP="${ALLOW_GROUP:-}"
if [[ -z "${ALLOW_GROUP}" ]]; then
  echo "Error: ALLOW_GROUP is required (e.g. ALLOW_GROUP=my-team@example.com). An IAP-secured gateway requires an authorized group or user." >&2
  exit 1
fi
GATEWAY_SA_NAME="dgemma-gateway-sa"
GATEWAY_SA="${GATEWAY_SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "================================================================"
echo " Deploying ${GATEWAY_SERVICE} (Go HTTP API & Web Studio Gateway)"
echo " Project: ${PROJECT} | Region: ${REGION}"
echo " Gateway SA: ${GATEWAY_SA} | Group: ${ALLOW_GROUP}"
echo "================================================================"

if ! gcloud iam service-accounts describe "${GATEWAY_SA}" --project="${PROJECT}" >/dev/null 2>&1; then
  echo "-> Creating Service Account ${GATEWAY_SA}..."
  gcloud iam service-accounts create "${GATEWAY_SA_NAME}" \
    --project="${PROJECT}" \
    --display-name="DiffusionGemma HTTP Gateway SA (Invoker on dgemma GPU)"
fi

UPSTREAM_URL="$(gcloud run services describe "${UPSTREAM_SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format='value(status.url)')"

if [ -z "${UPSTREAM_URL}" ]; then
  echo "ERROR: Upstream GPU service '${UPSTREAM_SERVICE}' not found in ${PROJECT}/${REGION}." >&2
  exit 1
fi
echo "-> Discovered Upstream GPU Service: ${UPSTREAM_URL}"

# Stage minimal build context (Go source + studio Lit app + templates) for fast ~20s Cloud Build
TMP_CTX="$(mktemp -d)"
trap 'rm -rf "${TMP_CTX}"' EXIT
cp go.mod go.sum main.go "${TMP_CTX}/"
cp -R cmd pkg templates "${TMP_CTX}/"
mkdir -p "${TMP_CTX}/studio"
cp studio/package*.json studio/tsconfig.json studio/vite.config.ts studio/index.html studio/embed.go "${TMP_CTX}/studio/"
cp -R studio/src "${TMP_CTX}/studio/"
mkdir -p "${TMP_CTX}/studio/dist"
touch "${TMP_CTX}/studio/dist/.gitkeep"
cp deploy/gateway/Dockerfile "${TMP_CTX}/Dockerfile"

echo "-> Building ${IMAGE} via Cloud Build..."
GW_VERSION="$(git describe --tags --match 'v*' --always --dirty 2>/dev/null || echo dev)"
cat > "${TMP_CTX}/cloudbuild.yaml" <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, --build-arg=DGEM_VERSION=${GW_VERSION}, --build-arg=DGEM_REVISION=$(git rev-parse --short HEAD), -t, "${IMAGE}", .]
images: ["${IMAGE}"]
YAML
# Submit asynchronously and poll: a synchronous submit exits non-zero when the caller cannot stream from the
# default logs bucket, even though the build succeeds.
BUILD_ID="$(gcloud builds submit "${TMP_CTX}" \
  --project="${PROJECT}" \
  --config="${TMP_CTX}/cloudbuild.yaml" \
  --async --format='value(id)' --quiet)"
echo "   build ${BUILD_ID}"
while true; do
  BUILD_STATUS="$(gcloud builds describe "${BUILD_ID}" --project="${PROJECT}" --format='value(status)')"
  case "${BUILD_STATUS}" in
    SUCCESS) break ;;
    QUEUED|WORKING|PENDING) sleep 10 ;;
    *) echo "ERROR: build ${BUILD_ID} ended with status ${BUILD_STATUS}" >&2; exit 1 ;;
  esac
done

GPU_IDLE_TTL="${GPU_IDLE_TTL:-3h}"
VERTEX_ENDPOINT_ID="${DGEM_VERTEX_URL:-}"
if [[ -z "${VERTEX_ENDPOINT_ID}" && "${ALLOW_NO_VERTEX:-0}" != "1" ]]; then
  echo "Error: DGEM_VERTEX_URL is required (e.g. DGEM_VERTEX_URL=<endpoint-id> or full /invoke/* URL). Set ALLOW_NO_VERTEX=1 to deploy Cloud Run-only." >&2
  exit 1
fi

ENV_VARS="UPSTREAM_DGEMMA_URL=${UPSTREAM_URL}/v1,DGEM_GCP_AUTH=1,DGEM_GPU_IDLE_TTL=${GPU_IDLE_TTL},DGEM_DEFAULT_BACKEND=${DEFAULT_BACKEND:-vertex_first}"
if [[ -n "${VERTEX_ENDPOINT_ID}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_VERTEX_URL=${VERTEX_ENDPOINT_ID}"
fi
if [[ -n "${DGEM_VERTEX_MODEL_ID:-}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_VERTEX_MODEL_ID=${DGEM_VERTEX_MODEL_ID}"
fi
if [[ -n "${DGEM_VERTEX_SA:-}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_VERTEX_SA=${DGEM_VERTEX_SA}"
fi
# Issue #1 controls: backend allow-list, extra selectable Vertex endpoints, and the admin API
# (gateway-wide settings + Vertex deploy/teardown from the Studio; off unless DGEM_ADMIN_API=1).
if [[ -n "${DGEM_BACKENDS:-}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_BACKENDS=${DGEM_BACKENDS//,/;}"
fi
if [[ -n "${DGEM_ALLOWED_VERTEX_ENDPOINTS:-}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_ALLOWED_VERTEX_ENDPOINTS=${DGEM_ALLOWED_VERTEX_ENDPOINTS//,/;}"
fi
if [[ "${DGEM_ADMIN_API:-0}" == "1" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_ADMIN_API=1"
fi
if [[ -n "${DGEM_GATEWAY_HOSTS:-}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_GATEWAY_HOSTS=${DGEM_GATEWAY_HOSTS}"
fi
if [[ -n "${GCP_PROJECT_NUMBER:-}" ]]; then
  ENV_VARS="${ENV_VARS},DGEM_GCP_PROJECT_NUMBER=${GCP_PROJECT_NUMBER}"
fi

# GATEWAY_TAG=<tag>: deploy as a tagged revision with no traffic (blue/green). Verify at
# https://<tag>---<service-url-host>, then: gcloud run services update-traffic <service> --to-tags=<tag>=100
TRAFFIC_FLAGS=()
if [[ -n "${GATEWAY_TAG:-}" ]]; then
  TRAFFIC_FLAGS=(--no-traffic "--tag=${GATEWAY_TAG}")
fi

echo "-> Deploying Cloud Run service ${GATEWAY_SERVICE} (GPU_IDLE_TTL=${GPU_IDLE_TTL}, DGEM_VERTEX_URL=${VERTEX_ENDPOINT_ID:-<none>}${GATEWAY_TAG:+, no-traffic tag ${GATEWAY_TAG}})..."
# Service-to-service key for the serving image (X-DGem-Key; see docs/deploy/public-images.md). Held only by services
# (gateway, matrix jobs, probes), never given to people. Skipped when the secret doesn't exist.
SERVER_KEY_SECRET="${SERVER_KEY_SECRET:-dgem-server-key}"
GATEWAY_SECRET_FLAGS=()
if gcloud secrets describe "${SERVER_KEY_SECRET}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud secrets add-iam-policy-binding "${SERVER_KEY_SECRET}" --project "${PROJECT}" \
    --member="serviceAccount:${GATEWAY_SA}" --role=roles/secretmanager.secretAccessor --quiet >/dev/null
  GATEWAY_SECRET_FLAGS=(--set-secrets="DGEM_SERVER_KEY=${SERVER_KEY_SECRET}:latest")
fi

gcloud run deploy "${GATEWAY_SERVICE}" "${TRAFFIC_FLAGS[@]}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --service-account="${GATEWAY_SA}" \
  --cpu=1 \
  --memory=512Mi \
  --min-instances=1 \
  --no-cpu-throttling \
  --max-instances=10 \
  --concurrency=80 \
  --timeout=600 \
  --no-allow-unauthenticated \
  --set-env-vars="${ENV_VARS}" \
  ${GATEWAY_SECRET_FLAGS[@]+"${GATEWAY_SECRET_FLAGS[@]}"} \
  --quiet

# Ensure Gateway SA has Vertex AI User role to query/invoke Dedicated Endpoint ${VERTEX_ENDPOINT_ID}
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${GATEWAY_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  --quiet >/dev/null 2>&1 || true

# Run Zero-Trust IAM & IAP bindings for dgemma-gpu-sa, dgemma-gateway-sa, and group:${ALLOW_GROUP}
GCP_PROJECT="${PROJECT}" GCP_REGION="${REGION}" ALLOW_GROUP="${ALLOW_GROUP}" ./scripts/setup_cloudrun_iam.sh

GATEWAY_URL="$(gcloud run services describe "${GATEWAY_SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format='value(status.url)')"

echo "================================================================"
echo " ✅ ${GATEWAY_SERVICE} deployed: ${GATEWAY_URL}"
echo " -> Upstream GPU Backend: ${UPSTREAM_URL}/v1"
echo " -> Authorized Group:     group:${ALLOW_GROUP}"
echo "================================================================"
