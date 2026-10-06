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

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi
DASHBOARD_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deploy/monitoring/dgem-operational-dashboard.json"

echo "================================================================================"
echo " Provisioning Cloud Logging Metrics & Operational Dashboard (${PROJECT_ID})"
echo "================================================================================"

upsert_log_metric() {
  local name="$1"
  local cfg_file="$2"
  if gcloud logging metrics describe "${name}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  -> Updating log-based metric: ${name}..."
    gcloud logging metrics update "${name}" --config-from-file="${cfg_file}" --project="${PROJECT_ID}" --quiet
  else
    echo "  -> Creating log-based metric: ${name}..."
    gcloud logging metrics create "${name}" --config-from-file="${cfg_file}" --project="${PROJECT_ID}" --quiet
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Counter Metric: dgem_decisions_total (with surface, template, user, multimodal, reads labels)
cat > "${TMP_DIR}/dgem_decisions_total.yaml" <<'EOF'
description: "Total DiffusionGemma decision evaluations across Web Studio, MCP, and REST API"
filter: 'resource.type="cloud_run_revision" AND jsonPayload.span_name="dgem.gateway.decide"'
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  unit: "1"
  labels:
    - key: surface
      valueType: STRING
      description: "Access surface (web_studio, mcp_agent, rest_api, openai_compat)"
    - key: template
      valueType: STRING
      description: "Decision policy template name (.json.tmpl)"
    - key: user
      valueType: STRING
      description: "Authenticated user email"
    - key: multimodal
      valueType: STRING
      description: "Whether SigLIP vision input was used (true/false)"
    - key: reads
      valueType: STRING
      description: "Forward passes (1 = single-pass O(1), 2+ = conditional DAG)"
labelExtractors:
  surface: 'EXTRACT(jsonPayload.dgem_surface)'
  template: 'EXTRACT(jsonPayload.dgem_template)'
  user: 'EXTRACT(jsonPayload.dgem_user)'
  multimodal: 'EXTRACT(jsonPayload.dgem_multimodal)'
  reads: 'EXTRACT(jsonPayload.dgem_gpu_reads)'
EOF
upsert_log_metric "dgem_decisions_total" "${TMP_DIR}/dgem_decisions_total.yaml"

# 2. Distribution Metric: dgem_epistemic_entropy (Shannon H_max in nats)
cat > "${TMP_DIR}/dgem_epistemic_entropy.yaml" <<'EOF'
description: "Calibrated Epistemic Shannon Entropy (H_max in nats) per DiffusionGemma decision"
filter: 'resource.type="cloud_run_revision" AND jsonPayload.span_name="dgem.gateway.decide" AND jsonPayload.dgem_max_entropy >= 0'
valueExtractor: 'EXTRACT(jsonPayload.dgem_max_entropy)'
bucketOptions:
  linearBuckets:
    numFiniteBuckets: 25
    width: 0.05
    offset: 0.0
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "1"
  labels:
    - key: template
      valueType: STRING
      description: "Decision policy template name"
    - key: surface
      valueType: STRING
      description: "Access surface"
labelExtractors:
  template: 'EXTRACT(jsonPayload.dgem_template)'
  surface: 'EXTRACT(jsonPayload.dgem_surface)'
EOF
upsert_log_metric "dgem_epistemic_entropy" "${TMP_DIR}/dgem_epistemic_entropy.yaml"

# 3. Distribution Metric: dgem_gpu_forward_ms (Pure GPU forward-pass duration in ms)
cat > "${TMP_DIR}/dgem_gpu_forward_ms.yaml" <<'EOF'
description: "Pure vLLM GPU forward-pass duration (ms) for DiffusionGemma decision policies"
filter: 'resource.type="cloud_run_revision" AND jsonPayload.span_name="dgem.gateway.decide" AND jsonPayload.dgem_gpu_forward_ms > 0'
valueExtractor: 'EXTRACT(jsonPayload.dgem_gpu_forward_ms)'
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 30
    growthFactor: 1.3
    scale: 20.0
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "ms"
  labels:
    - key: template
      valueType: STRING
      description: "Decision policy template name"
    - key: surface
      valueType: STRING
      description: "Access surface"
labelExtractors:
  template: 'EXTRACT(jsonPayload.dgem_template)'
  surface: 'EXTRACT(jsonPayload.dgem_surface)'
EOF
upsert_log_metric "dgem_gpu_forward_ms" "${TMP_DIR}/dgem_gpu_forward_ms.yaml"

# 3b. Distribution Metric: dgem_gpu_prefill_ms (GPU prompt + SigLIP prefill duration in ms)
cat > "${TMP_DIR}/dgem_gpu_prefill_ms.yaml" <<'EOF'
description: "vLLM GPU prompt and SigLIP vision prefill duration (ms) for DiffusionGemma decision policies"
filter: 'resource.type="cloud_run_revision" AND jsonPayload.span_name="dgem.gateway.decide" AND jsonPayload.dgem_gpu_prefill_ms > 0'
valueExtractor: 'EXTRACT(jsonPayload.dgem_gpu_prefill_ms)'
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 30
    growthFactor: 1.3
    scale: 10.0
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "ms"
  labels:
    - key: template
      valueType: STRING
      description: "Decision policy template name"
    - key: surface
      valueType: STRING
      description: "Access surface"
labelExtractors:
  template: 'EXTRACT(jsonPayload.dgem_template)'
  surface: 'EXTRACT(jsonPayload.dgem_surface)'
EOF
upsert_log_metric "dgem_gpu_prefill_ms" "${TMP_DIR}/dgem_gpu_prefill_ms.yaml"

# 3c. Distribution Metric: dgem_gpu_denoise_ms (Bidirectional diffusion canvas denoising duration in ms)
cat > "${TMP_DIR}/dgem_gpu_denoise_ms.yaml" <<'EOF'
description: "vLLM bidirectional diffusion canvas denoising duration (ms) for DiffusionGemma decision policies"
filter: 'resource.type="cloud_run_revision" AND jsonPayload.span_name="dgem.gateway.decide" AND jsonPayload.dgem_gpu_denoise_ms > 0'
valueExtractor: 'EXTRACT(jsonPayload.dgem_gpu_denoise_ms)'
bucketOptions:
  exponentialBuckets:
    numFiniteBuckets: 30
    growthFactor: 1.3
    scale: 15.0
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "ms"
  labels:
    - key: template
      valueType: STRING
      description: "Decision policy template name"
    - key: surface
      valueType: STRING
      description: "Access surface"
labelExtractors:
  template: 'EXTRACT(jsonPayload.dgem_template)'
  surface: 'EXTRACT(jsonPayload.dgem_surface)'
EOF
upsert_log_metric "dgem_gpu_denoise_ms" "${TMP_DIR}/dgem_gpu_denoise_ms.yaml"

# 4. Distribution Metric: dgem_warmup_total_ms (Cold-start GPU warmup duration in ms)
cat > "${TMP_DIR}/dgem_warmup_total_ms.yaml" <<'EOF'
description: "Total GPU cold-start warmup duration (ms) for right-sizing scale-from-zero"
filter: 'resource.type="cloud_run_revision" AND jsonPayload.span_name="dgem.gpu.warmup_lifecycle" AND jsonPayload.dgem_warmup_total_ms > 0'
valueExtractor: 'EXTRACT(jsonPayload.dgem_warmup_total_ms)'
bucketOptions:
  linearBuckets:
    numFiniteBuckets: 30
    width: 10000.0
    offset: 10000.0
metricDescriptor:
  metricKind: DELTA
  valueType: DISTRIBUTION
  unit: "ms"
  labels:
    - key: trigger
      valueType: STRING
      description: "Warmup trigger source (web_studio_wake, auto_wake_decide, mcp_agent, api_warmup)"
labelExtractors:
  trigger: 'EXTRACT(jsonPayload.dgem_warmup_trigger)'
EOF
upsert_log_metric "dgem_warmup_total_ms" "${TMP_DIR}/dgem_warmup_total_ms.yaml"

# 5. Create or Update Cloud Monitoring Operational Dashboard
EXISTING_DASHBOARD="$(gcloud monitoring dashboards list --project="${PROJECT_ID}" --filter='displayName="DiffusionGemma (dgem) — Operational & Decision Intelligence Dashboard"' --format='value(name)' | head -n1)"
if [[ -n "${EXISTING_DASHBOARD}" ]]; then
  echo "  -> Updating existing Cloud Monitoring dashboard (${EXISTING_DASHBOARD})..."
  ETAG="$(gcloud monitoring dashboards describe "${EXISTING_DASHBOARD}" --project="${PROJECT_ID}" --format='value(etag)')"
  jq --arg etag "${ETAG}" '. + {etag: $etag}' "${DASHBOARD_FILE}" > "${TMP_DIR}/dashboard_update.json"
  gcloud monitoring dashboards update "${EXISTING_DASHBOARD}" --config-from-file="${TMP_DIR}/dashboard_update.json" --project="${PROJECT_ID}" --quiet
  DASHBOARD_ID="${EXISTING_DASHBOARD##*/}"
else
  echo "  -> Creating new Cloud Monitoring dashboard..."
  CREATED_OUT="$(gcloud monitoring dashboards create --config-from-file="${DASHBOARD_FILE}" --project="${PROJECT_ID}" --format='value(name)')"
  DASHBOARD_ID="${CREATED_OUT##*/}"
fi

echo "================================================================================"
echo " Cloud Monitoring Dashboard Provisioned Successfully!"
echo " URL: https://console.cloud.google.com/monitoring/dashboards/builder/${DASHBOARD_ID}?project=${PROJECT_ID}"
echo "================================================================================"
