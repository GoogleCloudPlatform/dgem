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

# Stage DiffusionGemma Model Weights in Google Cloud Storage for Cloud Run GCS FUSE Mounting

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${2:-${GCP_REGION:-us-central1}}"
BUCKET="${1:-${GCS_BUCKET:-}}"
MODEL_ID="${3:-${MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi

if [[ -z "$BUCKET" ]]; then
  BUCKET="dgem-weights-${PROJECT_ID}"
fi

echo "================================================================================"
echo "  Staging DiffusionGemma Model Weights in GCS for Cloud Run GCS FUSE"
echo "================================================================================"
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Bucket:   gs://$BUCKET"
echo "Model:    $MODEL_ID"
echo "================================================================================"

# 1. Create regional bucket if it does not exist
if ! gcloud storage buckets describe "gs://${BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "==> Creating regional GCS bucket gs://${BUCKET} in ${REGION}..."
  gcloud storage buckets create "gs://${BUCKET}" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access
fi

# 2. Check if weights already staged
if gcloud storage ls "gs://${BUCKET}/dgemma/config.json" >/dev/null 2>&1; then
  echo "==> Weights already present at gs://${BUCKET}/dgemma/:"
  gcloud storage ls -lh "gs://${BUCKET}/dgemma/"
  exit 0
fi

# 3. Stage weights into GCS via Cloud Build (zero local disk footprint)
echo "==> Staging weights via Google Cloud Build directly to gs://${BUCKET}/dgemma/..."
BUILD_CONFIG=$(mktemp /tmp/cloudbuild-stage-XXXXXX)
mv "$BUILD_CONFIG" "${BUILD_CONFIG}.yaml"
BUILD_CONFIG="${BUILD_CONFIG}.yaml"
trap 'rm -f "$BUILD_CONFIG"' EXIT

# Read HF_TOKEN if available
HF_TOKEN_VAL=""
if [[ -f .env ]]; then
  HF_TOKEN_VAL=$(grep HF_TOKEN .env | cut -d= -f2- | tr -d '"\047' || true)
elif [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_VAL="$HF_TOKEN"
fi

cat <<EOF > "$BUILD_CONFIG"
steps:
- name: 'python:3.11-slim'
  entrypoint: 'bash'
  args:
  - '-c'
  - |
    pip install --no-cache-dir -U huggingface_hub
    python3 -c "
    import os, sys
    from huggingface_hub import snapshot_download
    token = os.environ.get('HF_TOKEN', None) or None
    print('Downloading ${MODEL_ID} directly in Cloud Build...')
    snapshot_download(repo_id='${MODEL_ID}', local_dir='/workspace/dgemma', token=token)
    "
  env:
  - "HF_TOKEN=${HF_TOKEN_VAL}"
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk:slim'
  entrypoint: 'gcloud'
  args:
  - 'storage'
  - 'cp'
  - '-r'
  - '/workspace/dgemma/*'
  - 'gs://${BUCKET}/dgemma/'
options:
  machineType: 'E2_HIGHCPU_8'
  diskSizeGb: 100
EOF

gcloud builds submit --no-source \
  --project="$PROJECT_ID" \
  --config="$BUILD_CONFIG" \
  --timeout=1200s

echo ""
echo "================================================================================"
echo "  Model Weights Successfully Staged in GCS!"
echo "================================================================================"
echo "GCS Path: gs://${BUCKET}/dgemma/"
gcloud storage ls -l --readable-sizes "gs://${BUCKET}/dgemma/"
echo ""
echo "Now deploy to Cloud Run:"
echo "  GCS_BUCKET=\"$BUCKET\" make cloudrun-deploy"
echo "================================================================================"
