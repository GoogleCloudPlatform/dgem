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

# ==============================================================================
# Deploy DiffusionGemma (dgemma) to a Vertex AI Dedicated Endpoint using
# Arbitrary Custom Routes (invokeRoutePrefix: "/*")
#
# Reference:
#   https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/use-arbitrary-custom-routes
#
# Exposed Custom Routes on the Dedicated Endpoint:
#   POST .../invoke/v1/chat/completions      -> structured_server.py (/v1/chat/completions)
#   POST .../invoke/v1/systemone             -> native SystemOne/JevBench route (/v1/systemone)
#   GET  .../invoke/health                   -> container & vLLM readiness telemetry (/health)
# ==============================================================================
set -euo pipefail

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi
PROJECT_NUMBER="${GCP_PROJECT_NUMBER:-$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)' 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/dgem/dgemma:latest}"
GCS_BUCKET="${GCS_BUCKET:-dgem-weights-${PROJECT_ID}}"
ARTIFACT_URI="${ARTIFACT_URI:-gs://${GCS_BUCKET}/dgemma}"

# Hardware profile (VERTEX_PROFILE):
#   l4             g2-standard-16 + 1x NVIDIA_L4 (24 GB VRAM, 64 GB RAM). SigLIP off by default (tight VRAM).
#                  Endpoint/model names: dgemma-dedicated / dgemma-invoke (the original endpoint).
#   g4-rtxpro6000  g4-standard-48 + 1x NVIDIA_RTX_PRO_6000 (Blackwell, native FP4, 180 GB host RAM).
#                  SigLIP on (multimodal parity with Cloud Run). Names: dgemma-dedicated-g4 / dgemma-invoke-g4.
# Any VERTEX_* / engine variable below still overrides the profile.
VERTEX_PROFILE="${VERTEX_PROFILE:-g4-rtxpro6000}"
case "$VERTEX_PROFILE" in
  l4)
    P_MACHINE=g2-standard-16; P_ACCEL=NVIDIA_L4; P_COUNT=1; P_MM=1; P_KV=2; P_SEQS=32; P_MAXREP=1
    P_EP=dgemma-dedicated; P_MODEL=dgemma-invoke ;;
  g4-rtxpro6000)
    P_MACHINE=g4-standard-48; P_ACCEL=NVIDIA_RTX_PRO_6000; P_COUNT=1; P_MM=0; P_KV=12; P_SEQS=32; P_MAXREP=2
    P_EP=dgemma-dedicated-g4; P_MODEL=dgemma-invoke-g4 ;;
  *) echo "Unknown VERTEX_PROFILE=$VERTEX_PROFILE (use l4 | g4-rtxpro6000)" >&2; exit 1 ;;
esac

MODEL_DISPLAY_NAME="${VERTEX_MODEL_NAME:-$P_MODEL}"
ENDPOINT_DISPLAY_NAME="${VERTEX_ENDPOINT_NAME:-$P_EP}"
MACHINE_TYPE="${VERTEX_MACHINE_TYPE:-$P_MACHINE}"
ACCELERATOR_TYPE="${VERTEX_ACCELERATOR_TYPE:-$P_ACCEL}"
ACCELERATOR_COUNT="${VERTEX_ACCELERATOR_COUNT:-$P_COUNT}"
DISABLE_MM="${DISABLE_MM:-$P_MM}"
MIN_REPLICAS="${VERTEX_MIN_REPLICAS:-1}"
MAX_REPLICAS="${VERTEX_MAX_REPLICAS:-$P_MAXREP}"
# Engine settings are passed explicitly so Vertex and Cloud Run can run identical configurations.
KV_CACHE_GB="${KV_CACHE_GB:-$P_KV}"
MAX_SEQS="${MAX_SEQS:-$P_SEQS}"
# Context length: 8192 on G4 (same short-prompt accuracy and latency as 4096; prompts up to 8k answered
# instead of refused). The Decision Index canary uses 32768. The L4 profile keeps 4096 (24 GB VRAM).
if [ "${VERTEX_PROFILE}" = "l4" ]; then DEFAULT_CTX=4096; else DEFAULT_CTX=8192; fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_CTX}"
# Container stdout/stderr to Cloud Logging. Off by default on these deployments, which hid an engine crash.
# This script calls v1beta1, where the field is enableContainerLogging (disableContainerLogging is v1-only and was
# silently ignored here).
ENABLE_CONTAINER_LOGGING="${ENABLE_CONTAINER_LOGGING:-true}"
CANVAS="${CANVAS:-128}"
DEFAULT_SAMPLES="${DEFAULT_SAMPLES:-1}"
MAX_INFLIGHT="${MAX_INFLIGHT:-8}"
# Scale out when average GPU duty cycle exceeds this percentage (only used when MAX_REPLICAS > MIN_REPLICAS).
AUTOSCALE_DUTY_CYCLE="${AUTOSCALE_DUTY_CYCLE:-70}"
SERVICE_ACCOUNT="${VERTEX_SERVICE_ACCOUNT:-dgemma-gpu-sa@${PROJECT_ID}.iam.gserviceaccount.com}"

API_BASE="https://${REGION}-aiplatform.googleapis.com/v1beta1"
TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null || gcloud auth print-access-token)"

echo "==> Profile ${VERTEX_PROFILE}: ${MACHINE_TYPE} + ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}, replicas ${MIN_REPLICAS}-${MAX_REPLICAS}, DISABLE_MM=${DISABLE_MM}, KV_CACHE_GB=${KV_CACHE_GB}, MAX_SEQS=${MAX_SEQS}, DEFAULT_SAMPLES=${DEFAULT_SAMPLES}"
echo "    Image: ${IMAGE_URI}"
echo "    Endpoint display name: ${ENDPOINT_DISPLAY_NAME} (an existing endpoint with this name is reused)"

echo "==> [1/4] Uploading Invoke-Enabled Model (${MODEL_DISPLAY_NAME}) with invokeRoutePrefix=\"/*\"..."
UPLOAD_PAYLOAD=$(cat <<EOF
{
  "model": {
    "displayName": "${MODEL_DISPLAY_NAME}",
    "containerSpec": {
      "imageUri": "${IMAGE_URI}",
      "invokeRoutePrefix": "/*",
      "healthRoute": "/health",
      "ports": [
        { "containerPort": 8080 }
      ],
      "env": [
        { "name": "DGEM_WEIGHTS_URI", "value": "${ARTIFACT_URI}" },
        { "name": "GCS_BUCKET", "value": "${GCS_BUCKET}" },
        { "name": "CANVAS", "value": "${CANVAS}" },
        { "name": "ENFORCE_EAGER", "value": "1" },
        { "name": "DISABLE_MM", "value": "${DISABLE_MM}" },
        { "name": "GPU_UTIL", "value": "0.85" },
        { "name": "KV_CACHE_GB", "value": "${KV_CACHE_GB}" },
        { "name": "MAX_SEQS", "value": "${MAX_SEQS}" },
        { "name": "MAX_MODEL_LEN", "value": "${MAX_MODEL_LEN}" },
        { "name": "HTTP_IDLE_TIMEOUT", "value": "${HTTP_IDLE_TIMEOUT:-120}" },
        { "name": "DEFAULT_SAMPLES", "value": "${DEFAULT_SAMPLES}" },
        { "name": "MAX_INFLIGHT", "value": "${MAX_INFLIGHT}" }
      ]
    }
  }
}
EOF
)

UPLOAD_OP=$(curl -sS -X POST "${API_BASE}/projects/${PROJECT_ID}/locations/${REGION}/models:upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "${UPLOAD_PAYLOAD}")

OP_NAME=$(echo "$UPLOAD_OP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name",""))')
if [ -z "$OP_NAME" ]; then
  echo "ERROR uploading model: $UPLOAD_OP" >&2
  exit 1
fi
echo "    Waiting for model upload operation: ${OP_NAME}..."
while true; do
  TOKEN="$(gcloud auth print-access-token)"
  OP_RES=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${OP_NAME}")
  DONE=$(echo "$OP_RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("done", False))')
  if [ "$DONE" = "True" ]; then
    MODEL_RESOURCE=$(echo "$OP_RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("response",{}).get("model",""))')
    break
  fi
  sleep 5
done
echo "    Uploaded Model Resource: ${MODEL_RESOURCE}"

echo "==> [2/4] Finding or Creating Dedicated Endpoint (${ENDPOINT_DISPLAY_NAME}, dedicatedEndpointEnabled=true)..."
EXISTING_EP=$(curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "${API_BASE}/projects/${PROJECT_ID}/locations/${REGION}/endpoints" | \
  python3 -c "import json,sys; eps=json.load(sys.stdin).get('endpoints',[]); print(next((e['name'] for e in eps if e.get('displayName')=='${ENDPOINT_DISPLAY_NAME}' and e.get('dedicatedEndpointEnabled')), ''))")

if [ -n "$EXISTING_EP" ]; then
  ENDPOINT_RESOURCE="$EXISTING_EP"
  echo "    Reusing existing Dedicated Endpoint: ${ENDPOINT_RESOURCE}"
else
  CREATE_EP_OP=$(curl -sS -X POST "${API_BASE}/projects/${PROJECT_ID}/locations/${REGION}/endpoints" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"displayName\": \"${ENDPOINT_DISPLAY_NAME}\", \"dedicatedEndpointEnabled\": true}")
  EP_OP_NAME=$(echo "$CREATE_EP_OP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name",""))')
  while true; do
    TOKEN="$(gcloud auth print-access-token)"
    OP_RES=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${EP_OP_NAME}")
    DONE=$(echo "$OP_RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("done", False))')
    if [ "$DONE" = "True" ]; then
      ENDPOINT_RESOURCE=$(echo "$OP_RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("response",{}).get("name",""))')
      break
    fi
    sleep 4
  done
  echo "    Created Dedicated Endpoint: ${ENDPOINT_RESOURCE}"
fi

ENDPOINT_ID="${ENDPOINT_RESOURCE##*/}"

AUTOSCALE_JSON=""
if [ "${MAX_REPLICAS}" -gt "${MIN_REPLICAS}" ]; then
  AUTOSCALE_JSON=",
      \"autoscalingMetricSpecs\": [{\"metricName\": \"aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle\", \"target\": ${AUTOSCALE_DUTY_CYCLE}}]"
fi

# VERTEX_NEW_TRAFFIC=0 deploys next to the models already on the endpoint without moving traffic (blue/green);
# shift traffic later with an endpoint trafficSplit update. Default: the new model takes 100%.
VERTEX_NEW_TRAFFIC="${VERTEX_NEW_TRAFFIC:-100}"
if [ "${VERTEX_NEW_TRAFFIC}" = "0" ]; then
  TRAFFIC_JSON=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${ENDPOINT_RESOURCE}" | python3 -c '
import json, sys
split = json.load(sys.stdin).get("trafficSplit") or {}
split["0"] = 0
print(json.dumps(split))')
else
  TRAFFIC_JSON='{"0": 100}'
fi

echo "==> [3/4] Deploying ${MODEL_RESOURCE} to ${ENDPOINT_RESOURCE} (${MACHINE_TYPE}, ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE})..."
DEPLOY_PAYLOAD=$(cat <<EOF
{
  "deployedModel": {
    "model": "${MODEL_RESOURCE}",
    "displayName": "${MODEL_DISPLAY_NAME}-deployment",
    "serviceAccount": "${SERVICE_ACCOUNT}",
    "enableContainerLogging": ${ENABLE_CONTAINER_LOGGING},
    "dedicatedResources": {
      "machineSpec": {
        "machineType": "${MACHINE_TYPE}",
        "acceleratorType": "${ACCELERATOR_TYPE}",
        "acceleratorCount": ${ACCELERATOR_COUNT}
      },
      "minReplicaCount": ${MIN_REPLICAS},
      "maxReplicaCount": ${MAX_REPLICAS}${AUTOSCALE_JSON}
    }
  },
  "trafficSplit": ${TRAFFIC_JSON}
}
EOF
)

DEPLOY_OP=$(curl -sS -X POST "${API_BASE}/${ENDPOINT_RESOURCE}:deployModel" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "${DEPLOY_PAYLOAD}")

DEPLOY_OP_NAME=$(echo "$DEPLOY_OP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name",""))')
if [ -z "$DEPLOY_OP_NAME" ]; then
  echo "ERROR deploying model: $DEPLOY_OP" >&2
  exit 1
fi
echo "    Deploy operation started: ${DEPLOY_OP_NAME}"
echo "    (Vertex AI Dedicated Endpoint provisioning + 17.53 GiB safetensors load typically takes 12-18 minutes.)"

while true; do
  TOKEN="$(gcloud auth print-access-token)"
  OP_RES=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${DEPLOY_OP_NAME}")
  DONE=$(echo "$OP_RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("done", False))')
  if [ "$DONE" = "True" ]; then
    break
  fi
  echo "    Still deploying (${ENDPOINT_ID})... sleeping 20s"
  sleep 20
done

echo "==> [4/4] Resolving Dedicated Endpoint DNS & Custom /invoke/* Routes..."
EP_INFO=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${ENDPOINT_RESOURCE}")
DEDICATED_DNS=$(echo "$EP_INFO" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("dedicatedEndpointDns",""))')
if [ -z "$DEDICATED_DNS" ]; then
  DEDICATED_DNS="${ENDPOINT_ID}.${REGION}-${PROJECT_NUMBER}.prediction.vertexai.goog"
fi

INVOKE_BASE_URL="https://${DEDICATED_DNS}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}/invoke"

cat <<EOF

================================================================================
✅ Vertex AI Dedicated Endpoint Deployed with Arbitrary Custom Routes (/*)
================================================================================
  Endpoint ID:            ${ENDPOINT_ID}
  Dedicated Endpoint DNS: ${DEDICATED_DNS}

  Mapped Custom Routes:
  • Structured Decision:  ${INVOKE_BASE_URL}/v1/chat/completions
  • SystemOne / JevBench: ${INVOKE_BASE_URL}/v1/systemone
  • Live Health Probe:    ${INVOKE_BASE_URL}/health

  Use with dgem CLI:
    ./bin/dgem decide -u "${INVOKE_BASE_URL}/v1" --gcp-auth -t templates/support_triage.json.tmpl -v ticket="Double billed"

  Use with dgemma-gateway (Header or Web Studio Settings):
    curl -sS "${GATEWAY_URL:-https://<your-dgem-gateway>}"/api/decide/support_triage \\
      -H "X-DGem-Backend: vertex" \\
      -H "X-DGem-Vertex-Url: ${ENDPOINT_ID}" \\
      -H "Content-Type: application/json" \\
      -d '{"variables":{"ticket":"My invoice was charged twice"}}'

  IMPORTANT (Zero-Idle-Cost Mandate):
  Vertex AI Dedicated Endpoints enforce minReplicaCount >= 1 (billed continuously).
  Always run 'make vertex-teardown' immediately after testing!
================================================================================
EOF
