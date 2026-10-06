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

echo "================================================================================"
echo "  Starting DiffusionGemma GCE Startup Script"
echo "================================================================================"

# Get HF_TOKEN from metadata if provided
HF_TOKEN=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token" -H "Metadata-Flavor: Google" || true)

# Wait for apt lock if necessary
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  echo "Waiting for package manager..."
  sleep 5
done

# Install vLLM nightly matching PR #57250 base commit
echo "==> Installing matching vLLM nightly wheel..."
pip install -U pip
pip install https://wheels.vllm.ai/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm-0.29.1rc1.dev410%2Bg36fa72d2d-cp38-abi3-manylinux_2_28_x86_64.whl huggingface_hub compressed-tensors
pip uninstall -y torchaudio

# Clone PR #57250 branch
echo "==> Cloning mmastrac/vllm structured-reads-main..."
rm -rf /tmp/vllm-pr
git clone --depth 1 -b structured-reads-main https://github.com/mmastrac/vllm.git /tmp/vllm-pr

VLLM_PATH=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "==> Overlaying PR #57250 files into $VLLM_PATH..."
for f in \
    model_executor/models/diffusion_gemma.py \
    sampling_params.py \
    v1/core/sched/diffusion_scheduler.py \
    v1/engine/input_processor.py \
    v1/worker/gpu/model_runner.py \
    v1/worker/gpu/model_states/interface.py \
    v1/worker/gpu/spec_decode/utils.py; do
  if [ -f "/tmp/vllm-pr/vllm/$f" ]; then
    mkdir -p "$VLLM_PATH/$(dirname "$f")"
    cp "/tmp/vllm-pr/vllm/$f" "$VLLM_PATH/$f"
  fi
done

mkdir -p /opt/diffusion_reads
cp -r /tmp/vllm-pr/examples/features/diffusion_reads/* /opt/diffusion_reads/

# Patch upstream vLLM multimodal runner bug where DiffusionGemma lacks _enable_mm_lora
sed -i 's/if self._enable_mm_lora:/if getattr(self, "_enable_mm_lora", False):/g' "$VLLM_PATH/model_executor/models/gemma4_mm.py" 2>/dev/null || true

# Find vllm executable path
VLLM_BIN=$(which vllm || echo "/usr/local/bin/vllm")

MODEL_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/model-id" -H "Metadata-Flavor: Google" 2>/dev/null || echo "nvidia/diffusiongemma-26B-A4B-it-NVFP4")
if [[ -z "$MODEL_ID" ]]; then
  MODEL_ID="nvidia/diffusiongemma-26B-A4B-it-NVFP4"
fi

TP_SIZE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/tensor-parallel-size" -H "Metadata-Flavor: Google" 2>/dev/null || echo "1")
if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="1"
fi

DTYPE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/dtype" -H "Metadata-Flavor: Google" 2>/dev/null || echo "auto")
if [[ -z "$DTYPE" ]]; then
  DTYPE="auto"
fi

# Create systemd service for vllm
echo "==> Creating systemd service for vllm ($MODEL_ID, TP=$TP_SIZE, dtype=$DTYPE)..."
cat <<EOF > /etc/systemd/system/vllm.service
[Unit]
Description=vLLM DiffusionGemma Service
After=network.target

[Service]
Type=simple
User=root
Environment="HF_TOKEN=$HF_TOKEN"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$VLLM_BIN serve $MODEL_ID \
  --tensor-parallel-size $TP_SIZE \
  --dtype $DTYPE \
  --diffusion-config '{"canvas_length":32}' \
  --max-model-len 32768 \
  --max-logprobs 32 \
  --enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --enforce-eager \
  --port 8080 \
  --host 0.0.0.0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vllm.service

echo "==> vLLM service started successfully."

BENCH_GCS_BUCKET=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bench-gcs-bucket" -H "Metadata-Flavor: Google" 2>/dev/null || true)
if [[ -n "$BENCH_GCS_BUCKET" ]]; then
  echo "==> Waiting for http://127.0.0.1:8080/health to run automated benchmark..."
  for i in {1..120}; do
    if curl -s -f "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
      echo "==> vLLM is healthy on localhost:8080!"
      break
    fi
    sleep 5
  done

  if curl -s -f "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    echo "==> Downloading benchmark bundle from $BENCH_GCS_BUCKET and running evaluation..."
    mkdir -p /opt/dgem-bench
    cd /opt/dgem-bench
    gcloud storage cp "${BENCH_GCS_BUCKET}/dgem-bench-bundle.tar.gz" . || gsutil cp "${BENCH_GCS_BUCKET}/dgem-bench-bundle.tar.gz" .
    tar -xzf dgem-bench-bundle.tar.gz
    chmod +x bin/dgem-linux
    ./bin/dgem-linux bench -u "http://127.0.0.1:8080/v1" -m "$MODEL_ID" -d benchmarks/eval_dataset.jsonl -M slot -o results_gce.json
  fi
fi
