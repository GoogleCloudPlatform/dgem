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

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
ZONE="${GCP_ZONE:-us-central1-a}"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

# Precision & Model Configuration
PRECISION="${PRECISION:-4}"
if [[ "$PRECISION" == "16" || "$PRECISION" == "unquant" || "$PRECISION" == "bfloat16" ]]; then
  DEFAULT_MODEL="google/diffusiongemma-26B-A4B-it"
  DEFAULT_MACHINE="a2-highgpu-2g"
  DEFAULT_TP="2"
  DEFAULT_DTYPE="bfloat16"
  DEFAULT_INSTANCE="diffgemma-bench-a100-2x"
elif [[ "$PRECISION" == "8" || "$PRECISION" == "fp8" || "$PRECISION" == "quant8" ]]; then
  DEFAULT_MODEL="RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"
  DEFAULT_MACHINE="a2-highgpu-1g"
  DEFAULT_TP="1"
  DEFAULT_DTYPE="auto"
  DEFAULT_INSTANCE="diffgemma-bench-fp8"
else
  DEFAULT_MODEL="nvidia/diffusiongemma-26B-A4B-it-NVFP4"
  DEFAULT_MACHINE="g2-standard-8"
  DEFAULT_TP="1"
  DEFAULT_DTYPE="auto"
  DEFAULT_INSTANCE="diffgemma-bench-l4"
fi

MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL}"
MACHINE_TYPE="${GCE_MACHINE_TYPE:-$DEFAULT_MACHINE}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TP}"
DTYPE="${DTYPE:-$DEFAULT_DTYPE}"
INSTANCE_NAME="${GCE_INSTANCE_NAME:-$DEFAULT_INSTANCE}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi

echo "================================================================================"
echo "  Deploying DiffusionGemma to Google Compute Engine"
echo "================================================================================"
echo "Project:       $PROJECT_ID"
echo "Zone:          $ZONE"
echo "Instance:      $INSTANCE_NAME"
echo "Machine Type:  $MACHINE_TYPE (TP=$TP_SIZE)"
echo "Model ID:      $MODEL_ID (precision: $PRECISION, dtype: $DTYPE)"
echo "Base Image:    $IMAGE_FAMILY ($IMAGE_PROJECT)"
echo "================================================================================"

# 1. Ensure firewall rule exists
if ! gcloud compute firewall-rules describe allow-diffgemma-8080 --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "==> Creating firewall rule allow-diffgemma-8080..."
  gcloud compute firewall-rules create allow-diffgemma-8080 \
    --project="$PROJECT_ID" \
    --allow=tcp:8080 \
    --target-tags=diffgemma-server \
    --description="Allow inbound traffic on port 8080 for DiffusionGemma benchmark"
fi

# 2. Check if instance already exists
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "==> Instance $INSTANCE_NAME already exists."
else
  echo "==> Step 1: Provisioning GCE GPU instance..."
  CREATE_FLAGS=(
    "--project=$PROJECT_ID"
    "--zone=$ZONE"
    "--machine-type=$MACHINE_TYPE"
    "--maintenance-policy=TERMINATE"
    "--image-family=$IMAGE_FAMILY"
    "--image-project=$IMAGE_PROJECT"
    "--boot-disk-size=250GB"
    "--boot-disk-type=pd-ssd"
    "--tags=diffgemma-server"
    "--metadata-from-file=startup-script=scripts/startup_gce_vllm.sh"
    "--metadata=hf-token=${HF_TOKEN:-},model-id=${MODEL_ID},tensor-parallel-size=${TP_SIZE},dtype=${DTYPE},install-nvidia-driver=True"
  )

  if [[ "$MACHINE_TYPE" == "g2-standard-16" ]]; then
    CREATE_FLAGS+=("--accelerator=type=nvidia-l4,count=2")
  elif [[ "$MACHINE_TYPE" =~ ^g2- ]]; then
    CREATE_FLAGS+=("--accelerator=type=nvidia-l4,count=1")
  fi

  gcloud compute instances create "$INSTANCE_NAME" "${CREATE_FLAGS[@]}"
fi

EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "Instance External IP: $EXTERNAL_IP"
echo "Target Endpoint:      http://${EXTERNAL_IP}:8080/v1"
echo ""
echo "==> Step 2: Waiting for vLLM DiffusionGemma to initialize on the VM..."
echo "(Startup script is installing vLLM and caching model weights. This takes ~3-5 minutes)"

START_TIME=$(date +%s)
TIMEOUT_SECS=900

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  if [[ $ELAPSED -gt $TIMEOUT_SECS ]]; then
    echo ""
    echo "Error: Timed out waiting for http://${EXTERNAL_IP}:8080/health after ${TIMEOUT_SECS}s."
    echo "Check serial console logs:"
    echo "  gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
    exit 1
  fi

  if curl -s -f "http://${EXTERNAL_IP}:8080/health" >/dev/null 2>&1 || curl -s -f "http://${EXTERNAL_IP}:8080/v1/models" >/dev/null 2>&1; then
    echo ""
    echo "================================================================================"
    echo "  vLLM DiffusionGemma is LIVE and Healthy!"
    echo "================================================================================"
    echo "Service URL: http://${EXTERNAL_IP}:8080/v1"
    echo ""
    echo "Run the benchmark evaluation:"
    echo "  ./bin/dgem bench -u \"http://${EXTERNAL_IP}:8080/v1\" \\"
    echo "    -m \"$MODEL_ID\" \\"
    echo "    -d benchmarks/eval_dataset.jsonl -M slot -o results_gce_${PRECISION}.json"
    echo ""
    echo "When finished, tear down the instance:"
    echo "  GCE_INSTANCE_NAME=\"$INSTANCE_NAME\" make gce-teardown"
    echo "================================================================================"
    break
  fi

  printf "."
  sleep 10
done
