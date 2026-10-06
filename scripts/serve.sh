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

MODEL_DIR="${DIFFGEMMA_MODEL_DIR:-model/diffgemma-26b-a4b-it-q4}"
CTX="${DIFFGEMMA_CTX:-32768}"
ADDR="${DIFFGEMMA_ADDR:-127.0.0.1:8080}"
PID_FILE="diffgemma.pid"
LOG_FILE="server.log"

if ! command -v diffgemma >/dev/null 2>&1; then
  echo "Error: diffgemma command not found on PATH."
  echo "Run ./scripts/setup_diffgemma.sh or 'make setup' first."
  exit 1
fi

if [[ ! -f "$MODEL_DIR/model.dgq.bin" ]]; then
  echo "Error: Model pack not found at $MODEL_DIR."
  echo "Run ./scripts/download_model.sh or 'make download' first."
  exit 1
fi

# Check if already running on that port
if lsof -i ":${ADDR##*:}" >/dev/null 2>&1; then
  echo "==> Server is already running or port ${ADDR##*:} is in use."
  curl -s "http://$ADDR/v1/models" || true
  exit 0
fi

echo "==> Launching diffgemma serve in background..."
echo "    Model: $MODEL_DIR"
echo "    Context: $CTX tokens"
echo "    Address: $ADDR"
echo "    Logs: $LOG_FILE"

nohup diffgemma serve -m "$MODEL_DIR" --ctx "$CTX" --addr "$ADDR" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "==> Process started with PID $SERVER_PID"

echo -n "==> Waiting for server to become healthy..."
for i in {1..30}; do
  if curl -s "http://$ADDR/v1/models" >/dev/null 2>&1; then
    echo " ready!"
    echo "==> Server listening on http://$ADDR"
    curl -s "http://$ADDR/v1/models"
    exit 0
  fi
  sleep 1
  echo -n "."
done

echo ""
echo "Warning: Server started but did not respond within 30 seconds."
echo "Check $LOG_FILE for details."
