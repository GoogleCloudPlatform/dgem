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

PID_FILE="dgem-gateway.pid"
GATEWAY_PORT="${DGEM_GATEWAY_PORT:-8090}"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "==> Stopping dgem gateway (PID $PID)..."
    kill "$PID" || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      echo "==> Force stopping PID $PID..."
      kill -9 "$PID" || true
    fi
  fi
  rm -f "$PID_FILE"
fi

if lsof -i ":$GATEWAY_PORT" >/dev/null 2>&1; then
  PORT_PIDS=$(lsof -t -i ":$GATEWAY_PORT" 2>/dev/null || true)
  if [[ -n "$PORT_PIDS" ]]; then
    echo "==> Releasing port $GATEWAY_PORT (PIDs: $PORT_PIDS)..."
    kill $PORT_PIDS 2>/dev/null || true
    sleep 1
    kill -9 $PORT_PIDS 2>/dev/null || true
  fi
fi

echo "==> dgem gateway stopped."
