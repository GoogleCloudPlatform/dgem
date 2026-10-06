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

# ==============================================================================
# Automated GPU Smoke Test with Guaranteed Immediate Teardown
# Supports Cloud Run GPU (default) or Vertex AI Dedicated Endpoint (--target vertex)
# Validates a dgem serving image you built (make cloudrun-build) on a short-lived GPU service.
# ==============================================================================

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
TARGET="${TARGET:-cloudrun}" # "cloudrun" or "vertex"
GPU_TYPE="${GPU_TYPE:-nvidia-rtx-pro-6000}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/dgem/dgemma:latest}"
SERVICE_NAME="dgem-smoke-$(date +%s)"
TEMPERATURE="${TEMPERATURE:-1.0}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"

# Portable millisecond clock (BSD/macOS `date` has no %N).
now_ms() { python3 -c 'import time; print(int(time.time() * 1000))'; }

# region_quota <service> <quota-id>: print the quota value that applies to $REGION
# ("0" if none/unset). The quota list covers every region and its order is not
# guaranteed, so pick the entry for $REGION instead of the first entry.
region_quota() {
  gcloud beta quotas info describe "$2" --service="$1" --project="$PROJECT_ID" --format=json 2>/dev/null \
    | REGION="$REGION" python3 -c '
import json, os, sys
region = os.environ["REGION"]
try:
    infos = json.load(sys.stdin).get("dimensionsInfos", [])
except Exception:
    print("0"); sys.exit()
exact = [i for i in infos if i.get("dimensions", {}).get("region") == region]
applies = [i for i in infos if region in i.get("applicableLocations", [])]
for i in exact or applies:
    print(i.get("details", {}).get("value") or "0"); break
else:
    print("0")
' || echo "0"
}

# health_ok <url> [bearer-token]: true if the endpoint returns HTTP 200 with {"status": "ok"}.
health_ok() {
  local url="$1" tok="${2:-}" body
  if [[ -n "$tok" ]]; then
    body=$(curl -sf -m 5 -H "Authorization: Bearer $tok" "$url") || return 1
  else
    body=$(curl -sf -m 5 "$url") || return 1
  fi
  printf '%s' "$body" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("status") == "ok" else 1)' 2>/dev/null
}

# Parse optional CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --keep-alive)
      KEEP_ALIVE=1
      shift
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --image)
      IMAGE_URI="$2"
      shift 2
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP hosting project specified."
  echo "Usage: GCP_PROJECT=<hosting-project> [TARGET=cloudrun|vertex] ./scripts/gpu_smoketest.sh [--target vertex] [--keep-alive]"
  exit 1
fi

echo "================================================================================"
echo "  dgem GPU Smoke Test (Target: $TARGET)"
echo "================================================================================"
echo "  Hosting Project:  $PROJECT_ID"
echo "  Region:           $REGION"
echo "  Target Surface:   $TARGET"
echo "  GPU Accelerator:  $GPU_TYPE"
echo "  Container Image:  $IMAGE_URI"
echo "  Keep-Alive:       $KEEP_ALIVE"
echo "================================================================================"

# ==============================================================================
# TARGET: VERTEX AI DEDICATED ENDPOINT
# ==============================================================================
if [[ "$TARGET" == "vertex" ]]; then
  # 1. Quota Check on aiplatform.googleapis.com
  VERTEX_QUOTA_ID="CustomModelServingRTXPRO6000GPUsPerProjectPerRegion"
  if [[ "$GPU_TYPE" == *"l4"* ]]; then
    VERTEX_QUOTA_ID="CustomModelServingL4GPUsPerProjectPerRegion"
  fi

  echo "==> Verifying Vertex AI GPU quota ($VERTEX_QUOTA_ID) in $PROJECT_ID ($REGION)..."
  QUOTA_VAL=$(region_quota aiplatform.googleapis.com "$VERTEX_QUOTA_ID")

  if [[ "$QUOTA_VAL" == "0" || "$QUOTA_VAL" == "null" || -z "$QUOTA_VAL" ]]; then
    echo ""
    echo "⚠️  Quota Check Failed: $PROJECT_ID has 0 quota for $VERTEX_QUOTA_ID in $REGION."
    echo "   Please request quota via Google Cloud Console:"
    echo "   https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}&service=aiplatform.googleapis.com"
    exit 2
  fi
  echo "✓ Active Vertex AI quota confirmed: $QUOTA_VAL GPU(s) in $REGION."

  EP_NAME="dgem-smoke-ep-$(date +%s)"
  MODEL_NAME="dgem-smoke-model-$(date +%s)"

  cleanup_vertex() {
    if [[ "$KEEP_ALIVE" == "1" ]]; then
      echo ""
      echo "KEEP_ALIVE=1: Preserving Vertex AI Dedicated Endpoint for benchmark runs."
      echo "To teardown later: GCP_PROJECT=$PROJECT_ID make vertex-teardown"
      return
    fi
    echo ""
    echo "================================================================================"
    echo "  MANDATORY TEARDOWN: Deleting ephemeral Vertex AI endpoint and model..."
    echo "================================================================================"
    EP_NUM=$(gcloud ai endpoints list --project="$PROJECT_ID" --region="$REGION" --filter="displayName=$EP_NAME" --format="value(name)" 2>/dev/null | head -1 || true)
    if [[ -n "$EP_NUM" ]]; then
      for DM in $(gcloud ai endpoints describe "$EP_NUM" --project="$PROJECT_ID" --region="$REGION" --format="value(deployedModels.id)" 2>/dev/null); do
        echo "-> Undeploying model $DM from endpoint $EP_NUM..."
        gcloud ai endpoints undeploy-model "$EP_NUM" --project="$PROJECT_ID" --region="$REGION" --deployed-model-id="$DM" --quiet 2>/dev/null || true
      done
      echo "-> Deleting endpoint $EP_NUM..."
      gcloud ai endpoints delete "$EP_NUM" --project="$PROJECT_ID" --region="$REGION" --quiet 2>/dev/null || true
    fi
    MOD_NUM=$(gcloud ai models list --project="$PROJECT_ID" --region="$REGION" --filter="displayName=$MODEL_NAME" --format="value(name)" 2>/dev/null | head -1 || true)
    if [[ -n "$MOD_NUM" ]]; then
      echo "-> Deleting model $MOD_NUM..."
      gcloud ai models delete "$MOD_NUM" --project="$PROJECT_ID" --region="$REGION" --quiet 2>/dev/null || true
    fi
    echo "✓ Ephemeral Vertex AI resources deleted. Zero idle cost."
  }
  trap cleanup_vertex EXIT INT TERM

  echo "==> Provisioning Vertex AI Dedicated Endpoint ($EP_NAME)..."
  PROFILE="g4-rtxpro6000"
  if [[ "$GPU_TYPE" == *"l4"* ]]; then
    PROFILE="l4"
  fi

  GCP_PROJECT="$PROJECT_ID" \
  GCP_REGION="$REGION" \
  VERTEX_ENDPOINT_NAME="$EP_NAME" \
  VERTEX_MODEL_NAME="$MODEL_NAME" \
  VERTEX_PROFILE="$PROFILE" \
  IMAGE_URI="$IMAGE_URI" \
  ./scripts/deploy_vertex_endpoint.sh

  EP_ID=$(gcloud ai endpoints list --project="$PROJECT_ID" --region="$REGION" --filter="displayName=$EP_NAME" --format="value(name)" | head -1)
  PROJ_NUM=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
  DEDICATED_HOST="${EP_ID}.${REGION}-${PROJ_NUM}.prediction.vertexai.goog"
  HEALTH_URL="https://${DEDICATED_HOST}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${EP_ID}/invoke/health"
  SYSTEMONE_URL="https://${DEDICATED_HOST}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${EP_ID}/invoke/v1/systemone"

  echo "==> Dedicated DNS: $DEDICATED_HOST"
  TOK=$(gcloud auth print-access-token)

  echo -n "==> Waiting for /invoke/health probe..."
  HEALTHY=0
  for i in {1..60}; do
    if health_ok "$HEALTH_URL" "$TOK"; then
      echo " healthy!"
      HEALTHY=1
      break
    fi
    sleep 3
    echo -n "."
  done
  if [[ "$HEALTHY" != "1" ]]; then
    echo ""
    echo "❌ /invoke/health did not return {\"status\": \"ok\"} within ~3 minutes." >&2
    exit 3
  fi

  echo "==> Executing test decision via POST /invoke/v1/systemone..."
  TEST_PAYLOAD='{
    "state": "Customer was double billed on transaction #48291.",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is this an urgent issue?"},
      "department": {
        "type": "choice",
        "instructions": "Route to the responsible department",
        "criteria": {
          "billing": "Invoice and charge disputes",
          "engineering": "Platform and bug reports",
          "support": "General customer inquiries"
        }
      }
    }
  }'
  START_T=$(now_ms)
  RESP=$(curl -sS -X POST "$SYSTEMONE_URL" \
    -H "Authorization: Bearer $TOK" \
    -H "Content-Type: application/json" \
    -d "$TEST_PAYLOAD")
  END_T=$(now_ms)
  WALL_MS=$(( END_T - START_T ))

  echo ""
  echo "=== Smoke Test Output (${WALL_MS} ms) ==="
  echo "$RESP" | python3 -m json.tool || echo "$RESP"
  echo ""

  CHOICE=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["answers"]["department"]["choice"])' 2>/dev/null || echo "")
  if [[ "$CHOICE" == "billing" ]]; then
    echo "🎉 VERTEX AI SMOKE TEST PASSED: Department routed correctly to 'billing'!"
  else
    echo "⚠️  Smoke test unexpected output: $RESP"
  fi
  exit 0
fi

# ==============================================================================
# TARGET: CLOUD RUN GPU (DEFAULT)
# ==============================================================================

# 1. Quota Pre-Check
QUOTA_ID="NvidiaRtxPro6000GpuAllocNoZonalRedundancyPerProjectRegion"
if [[ "$GPU_TYPE" == *"l4"* ]]; then
  QUOTA_ID="NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion"
fi

echo "==> Verifying Cloud Run GPU quota ($QUOTA_ID) in $PROJECT_ID ($REGION)..."
QUOTA_VAL=$(region_quota run.googleapis.com "$QUOTA_ID")

if [[ "$QUOTA_VAL" == "0" || "$QUOTA_VAL" == "null" || -z "$QUOTA_VAL" ]]; then
  echo ""
  echo "⚠️  Quota Check Failed: $PROJECT_ID has 0 quota for $GPU_TYPE in $REGION."
  echo "   Please request quota via Google Cloud Console before running GPU smoke tests:"
  echo "   https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}&service=run.googleapis.com"
  exit 2
fi

# Cloud Run reports GPU quota in thousandths of a GPU (e.g. 4000 = 4 GPUs).
echo "✓ Active Cloud Run quota confirmed: $QUOTA_VAL (~$(( QUOTA_VAL / 1000 )) GPU(s)) in $REGION."

# 2. Register Guaranteed Teardown Trap
cleanup() {
  if [[ "$KEEP_ALIVE" == "1" ]]; then
    echo ""
    echo "KEEP_ALIVE=1: Preserving Cloud Run service $SERVICE_NAME for benchmark runs."
    return
  fi
  echo ""
  echo "================================================================================"
  echo "  MANDATORY TEARDOWN: Deleting ephemeral Cloud Run service $SERVICE_NAME..."
  echo "================================================================================"
  gcloud run services delete "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --quiet 2>/dev/null || true
  echo "✓ Service $SERVICE_NAME deleted. Zero idle cost."
}
trap cleanup EXIT INT TERM

# 3. Deploy Ephemeral Cloud Run Service
MEMORY="80Gi"
CPU="8"
if [[ "$GPU_TYPE" == *"l4"* ]]; then
  MEMORY="32Gi"
  CPU="4"
fi

echo "==> Deploying ephemeral service $SERVICE_NAME to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --image="$IMAGE_URI" \
  --gpu=1 \
  --gpu-type="$GPU_TYPE" \
  --memory="$MEMORY" \
  --cpu="$CPU" \
  --no-cpu-throttling \
  --min-instances=0 \
  --max-instances=1 \
  --timeout=600 \
  --set-env-vars="ROLE=decision-index,TEMPERATURE=${TEMPERATURE},CANVAS=128" \
  --allow-unauthenticated

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" --region="$REGION" --format="value(status.url)")
echo "==> Service deployed at $SERVICE_URL"

# 4. Probe Health
echo -n "==> Waiting for /health probe..."
HEALTHY=0
for i in {1..90}; do
  if health_ok "${SERVICE_URL}/health"; then
    echo " healthy!"
    HEALTHY=1
    break
  fi
  sleep 2
  echo -n "."
done
if [[ "$HEALTHY" != "1" ]]; then
  echo ""
  echo "❌ /health did not return {\"status\": \"ok\"} within ~3 minutes." >&2
  exit 3
fi

# 5. Execute Test Decision via POST /v1/systemone
echo "==> Executing test decision via POST /v1/systemone..."
START_T=$(now_ms)
TEST_PAYLOAD='{
  "state": "Customer was double billed on transaction #48291.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Is this an urgent issue?"},
    "department": {
      "type": "choice",
      "instructions": "Route to the responsible department",
      "criteria": {
        "billing": "Invoice and charge disputes",
        "engineering": "Platform and bug reports",
        "support": "General customer inquiries"
      }
    }
  }
}'

RESP=$(curl -sS -X POST "${SERVICE_URL}/v1/systemone" \
  -H "Content-Type: application/json" \
  -d "$TEST_PAYLOAD")

END_T=$(now_ms)
WALL_MS=$(( END_T - START_T ))

echo ""
echo "=== Smoke Test Output (${WALL_MS} ms) ==="
echo "$RESP" | python3 -m json.tool || echo "$RESP"
echo ""

CHOICE=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["answers"]["department"]["choice"])' 2>/dev/null || echo "")
if [[ "$CHOICE" == "billing" ]]; then
  echo "🎉 SMOKE TEST PASSED: Department routed correctly to 'billing'!"
else
  echo "⚠️  Smoke test unexpected output: $RESP"
fi
