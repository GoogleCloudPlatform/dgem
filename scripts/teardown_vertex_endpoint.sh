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
# Teardown Vertex AI Dedicated Endpoint (dgemma-dedicated) & Undeploy Replicas
# Enforces the Zero-Idle-Cost Mandate so no Vertex AI GPU replicas remain active.
# ==============================================================================
set -euo pipefail

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi
REGION="${GCP_REGION:-us-central1}"
ENDPOINT_DISPLAY_NAME="${VERTEX_ENDPOINT_NAME:-dgemma-dedicated-g4}"  # the L4 profile used dgemma-dedicated
API_BASE="https://${REGION}-aiplatform.googleapis.com/v1beta1"
TOKEN="$(gcloud auth print-access-token)"

echo "==> Checking for active Vertex AI Endpoints matching '${ENDPOINT_DISPLAY_NAME}' in ${PROJECT_ID} (${REGION})..."
ENDPOINTS_JSON=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/projects/${PROJECT_ID}/locations/${REGION}/endpoints")

MATCHING_EPS=$(echo "$ENDPOINTS_JSON" | python3 -c "
import json, sys
eps = json.load(sys.stdin).get('endpoints', [])
for e in eps:
    if e.get('displayName') == '${ENDPOINT_DISPLAY_NAME}' or '${ENDPOINT_DISPLAY_NAME}' in e.get('name', ''):
        dm_ids = [dm.get('id') for dm in e.get('deployedModels', []) if dm.get('id')]
        print(e['name'] + '|' + ','.join(dm_ids))
")

if [ -z "$MATCHING_EPS" ]; then
  echo "✅ Zero active '${ENDPOINT_DISPLAY_NAME}' Vertex AI Endpoints found. Zero idle GPU billing."
  exit 0
fi

for ENTRY in $MATCHING_EPS; do
  EP_NAME="${ENTRY%%|*}"
  DM_LIST="${ENTRY#*|}"
  echo "==> Found Endpoint: ${EP_NAME}"
  if [ -n "$DM_LIST" ]; then
    IFS=',' read -ra DMS <<< "$DM_LIST"
    for DM_ID in "${DMS[@]}"; do
      echo "    Undeploying DeployedModel ID ${DM_ID} from ${EP_NAME}..."
      UNDEPLOY_OP=$(curl -sS -X POST "${API_BASE}/${EP_NAME}:undeployModel" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"deployedModelId\": \"${DM_ID}\"}")
      OP_NAME=$(echo "$UNDEPLOY_OP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name",""))')
      if [ -n "$OP_NAME" ]; then
        while true; do
          TOKEN="$(gcloud auth print-access-token)"
          OP_RES=$(curl -sS -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${OP_NAME}")
          DONE=$(echo "$OP_RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("done", False))')
          [ "$DONE" = "True" ] && break
          sleep 4
        done
      fi
    done
  fi
  echo "    Deleting Dedicated Endpoint ${EP_NAME}..."
  curl -sS -X DELETE -H "Authorization: Bearer ${TOKEN}" "${API_BASE}/${EP_NAME}" >/dev/null
  echo "    ✅ Deleted ${EP_NAME}."
done

echo "✅ Vertex AI Dedicated Endpoint teardown complete (0 GPU replicas running)."
