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
INSTANCE_NAME="${GCE_INSTANCE_NAME:-diffgemma-bench-l4}"

echo "================================================================================"
echo "  Tearing down GCE GPU benchmark infrastructure..."
echo "================================================================================"

if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "==> Deleting GCE instance $INSTANCE_NAME..."
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet
  echo "Instance deleted."
else
  echo "Instance $INSTANCE_NAME already deleted."
fi

if gcloud compute firewall-rules describe allow-diffgemma-8080 --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "==> Deleting firewall rule allow-diffgemma-8080..."
  gcloud compute firewall-rules delete allow-diffgemma-8080 --project="$PROJECT_ID" --quiet
  echo "Firewall rule deleted."
fi

echo "================================================================================"
echo "  Teardown Complete. Zero idle cost."
echo "================================================================================"
