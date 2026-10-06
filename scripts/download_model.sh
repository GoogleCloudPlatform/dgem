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

JOBS="${1:-8}"
MODEL_DIR="model/diffgemma-26b-a4b-it-q4"

if ! command -v diffgemma >/dev/null 2>&1; then
  echo "Error: diffgemma command not found on PATH."
  echo "Run ./scripts/setup_diffgemma.sh or 'make setup' first."
  exit 1
fi

echo "==> Downloading DiffusionGemma 4-bit pack (mmastrac/diffgemma-26b-a4b-it-q4) with $JOBS concurrent jobs..."
diffgemma download --jobs "$JOBS"

if [[ -f "$MODEL_DIR/model.dgq.bin" ]]; then
  echo "==> Model pack downloaded and verified: $MODEL_DIR"
  ls -lh "$MODEL_DIR"
else
  echo "Error: Model file $MODEL_DIR/model.dgq.bin missing after download."
  exit 1
fi
