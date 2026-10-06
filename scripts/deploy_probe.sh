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

# Deploy the dgem scheduled health check (synthetic probe): two Cloud Run jobs + two Cloud Scheduler triggers.
#
#   dgem-probe-hourly  every hour: Vertex endpoint (contract + latency) and the gateway (vertex_first route)
#   dgem-probe-daily   once a day: the scale-to-zero Cloud Run service (wakes the GPU once, ~20 GPU-minutes/day)
#
# Each run writes one JSON log line per target (probe_event="dgem.probe"); scripts/setup_alerts.py turns those into
# alerts. See docs/operate/monitoring.md.
#
# Required:
#   GCP_PROJECT            hosting project
#   PROBE_VERTEX_URL       Vertex dedicated invoke base: https://<ID>.<REGION>-<NUM>.prediction.vertexai.goog/v1/projects/<P>/locations/<R>/endpoints/<ID>/invoke
#   PROBE_GATEWAY_URL      gateway run.app URL (https://dgemma-gateway-<hash>-uc.a.run.app), not an IAP custom domain
#   PROBE_CLOUDRUN_URL     serving Cloud Run URL (https://dgemma-<hash>-uc.a.run.app)
# Optional:
#   GCP_REGION (us-central1)  HOURLY_SCHEDULE ("7 * * * *")  DAILY_SCHEDULE ("17 6 * * *")  SCHEDULE_TZ (Etc/UTC)
#   GATEWAY_SERVICE (dgemma-gateway)  CLOUDRUN_SERVICE (dgemma)  PROBE_SA_NAME (dgem-probe-sa)
#   PROBE_LATENCY_REQUESTS (20)  PROBE_DAILY=off (skip the Cloud Run job)
#   PROBE_GATEWAY_AUDIENCE  required when the gateway has IAP enabled: one of its programmatic OAuth clients
#                           (gcloud iap settings get --resource-type=cloud-run --service=<gateway> --region=<r>)
set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
: "${PROBE_VERTEX_URL:?set PROBE_VERTEX_URL (Vertex dedicated invoke base)}"
: "${PROBE_GATEWAY_URL:?set PROBE_GATEWAY_URL (gateway run.app URL)}"
PROBE_CLOUDRUN_URL="${PROBE_CLOUDRUN_URL:-}"
SA_NAME="${PROBE_SA_NAME:-dgem-probe-sa}"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
GATEWAY_SERVICE="${GATEWAY_SERVICE:-dgemma-gateway}"
CLOUDRUN_SERVICE="${CLOUDRUN_SERVICE:-dgemma}"
HOURLY_SCHEDULE="${HOURLY_SCHEDULE:-7 * * * *}"
DAILY_SCHEDULE="${DAILY_SCHEDULE:-17 6 * * *}"
SCHEDULE_TZ="${SCHEDULE_TZ:-Etc/UTC}"
N="${PROBE_LATENCY_REQUESTS:-20}"
TAG="$(git rev-parse --short HEAD 2>/dev/null || echo latest)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/dgem/dgem-probe:${TAG}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Service account ${SA}"
gcloud iam service-accounts describe "${SA}" --project="${PROJECT}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SA_NAME}" --project="${PROJECT}" \
    --display-name="dgem scheduled health check (read-only probe)"
# Call the Vertex endpoint (predict/invoke) and invoke the gateway and Cloud Run services. Nothing else.
gcloud projects add-iam-policy-binding "${PROJECT}" --member="serviceAccount:${SA}" \
  --role="roles/aiplatform.user" --condition=None --quiet >/dev/null
if [[ -n "${PROBE_GATEWAY_AUDIENCE:-}" ]]; then
  # IAP-protected gateway: the SA must also be an IAP-secured Web App User on it.
  gcloud iap web add-iam-policy-binding --project="${PROJECT}" --resource-type=cloud-run \
    --service="${GATEWAY_SERVICE}" --region="${REGION}" --member="serviceAccount:${SA}" \
    --role="roles/iap.httpsResourceAccessor" --condition=None --quiet >/dev/null
fi
for svc in "${GATEWAY_SERVICE}" "${CLOUDRUN_SERVICE}"; do
  gcloud run services add-iam-policy-binding "${svc}" --project="${PROJECT}" --region="${REGION}" \
    --member="serviceAccount:${SA}" --role="roles/run.invoker" --quiet >/dev/null
done

echo "==> Building ${IMAGE}"
CTX="$(mktemp -d)"; trap 'rm -rf "${CTX}"' EXIT
mkdir -p "${CTX}/scripts" "${CTX}/fixtures"
cp "${REPO_ROOT}"/scripts/{probe.py,contract_diff.py,serving_speed.py} "${CTX}/scripts/"
cp -R "${REPO_ROOT}/fixtures/bbox" "${CTX}/fixtures/"
cp "${REPO_ROOT}/deploy/probe/Dockerfile" "${CTX}/Dockerfile"
gcloud builds submit "${CTX}" --project="${PROJECT}" --tag="${IMAGE}" --suppress-logs --quiet

deploy_job() {  # name targets task-timeout
  local job="$1" targets="$2" timeout="$3"
  local extra="PROBE_LATENCY_REQUESTS=${N}"
  [[ -n "${PROBE_GATEWAY_AUDIENCE:-}" ]] && extra="${extra}|PROBE_GATEWAY_AUDIENCE=${PROBE_GATEWAY_AUDIENCE}"
  echo "==> Cloud Run job ${job} (${targets%%=*}...)"
  gcloud run jobs deploy "${job}" --project="${PROJECT}" --region="${REGION}" --image="${IMAGE}" \
    --service-account="${SA}" --tasks=1 --max-retries=0 --task-timeout="${timeout}" --cpu=1 --memory=512Mi \
    --set-env-vars="^|^PROBE_TARGETS=${targets}|${extra}" --quiet >/dev/null
  # Scheduler calls the Run Admin API as the probe SA (needs run.jobs.run on the job).
  gcloud run jobs add-iam-policy-binding "${job}" --project="${PROJECT}" --region="${REGION}" \
    --member="serviceAccount:${SA}" --role="roles/run.invoker" --quiet >/dev/null
}

schedule() {  # scheduler-name job cron
  local name="$1" job="$2" cron="$3"
  local uri="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${job}:run"
  local args=(--project="${PROJECT}" --location="${REGION}" --schedule="${cron}" --time-zone="${SCHEDULE_TZ}"
              --uri="${uri}" --http-method=POST --oauth-service-account-email="${SA}")
  if gcloud scheduler jobs describe "${name}" --project="${PROJECT}" --location="${REGION}" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "${name}" "${args[@]}" --quiet >/dev/null
  else
    gcloud scheduler jobs create http "${name}" "${args[@]}" --quiet >/dev/null
  fi
  echo "    scheduled ${name}: '${cron}' (${SCHEDULE_TZ})"
}

deploy_job dgem-probe-hourly "vertex=${PROBE_VERTEX_URL},gateway=${PROBE_GATEWAY_URL}" 600s
schedule dgem-probe-hourly dgem-probe-hourly "${HOURLY_SCHEDULE}"
if [[ "${PROBE_DAILY:-on}" != "off" && -n "${PROBE_CLOUDRUN_URL}" ]]; then
  # Long timeout: the first request waits for the scale-to-zero service to wake (~2.5 min).
  deploy_job dgem-probe-daily "cloudrun=${PROBE_CLOUDRUN_URL}" 1200s
  schedule dgem-probe-daily dgem-probe-daily "${DAILY_SCHEDULE}"
fi

cat <<EOF

Done. Run once now:
  gcloud run jobs execute dgem-probe-hourly --project=${PROJECT} --region=${REGION} --wait
Results: Cloud Logging, jsonPayload.probe_event="dgem.probe". Alerts: scripts/setup_alerts.py.
EOF
