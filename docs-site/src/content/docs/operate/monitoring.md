---
title: "Monitoring and Alerts"
description: "How to monitor a dgem deployment: log-based metrics and the operational dashboard, the scheduled health check, recommended alert policies and what to do when each fires."
---

`dgem` monitoring has three parts:

| Part | What it does | Set up by |
| :--- | :--- | :--- |
| **Log-based metrics and dashboard** | Turn gateway decision logs into metrics (decision count, hesitation, GPU forward / prefill / denoise time, warmup) and a Cloud Monitoring dashboard | `scripts/setup_cloud_monitoring.sh` (`make cloudrun-dashboard`) |
| **Scheduled health check** | Every hour, checks the Vertex endpoint and the gateway; once a day, the Cloud Run service | `scripts/deploy_probe.sh` |
| **Alert policies** | Notify you when something is broken or slow | You, in Cloud Monitoring, from the signals below |

The scripts are idempotent, so re-running them after a change updates the existing resources. Placeholders:
`<PROJECT>`, `<REGION>`, `<ENDPOINT_ID>`, `<EMAIL>`.

## Log-based metrics and the dashboard

```bash
GCP_PROJECT=<PROJECT> ./scripts/setup_cloud_monitoring.sh
```

It creates or updates these log-based metrics from the gateway's `dgem.gateway.decide` span logs, and the dashboard
in `deploy/monitoring/dgem-operational-dashboard.json`:

| Metric | What it counts or measures |
| :--- | :--- |
| `dgem_decisions_total` | Decisions, labelled by surface (Studio, MCP, REST, OpenAI-compatible), template, user, multimodal and reads |
| `dgem_epistemic_entropy` | Hesitation (entropy) of decisions |
| `dgem_gpu_forward_ms`, `dgem_gpu_prefill_ms`, `dgem_gpu_denoise_ms` | GPU time per decision: total forward pass, prompt (and image) prefill, canvas denoising |
| `dgem_warmup_total_ms` | Server self-warmup time at start-up |

Each gateway decision log line (`jsonPayload.event_type="dgem.decision.completed"`) also carries `dgem_backend`
(where it ran), `dgem_backend_requested`, `dgem_template` and `dgem_total_wall_ms`, which you can use for further
log-based metrics, for example the failover share below. See [Observability](/dgem/operate/observability/) for the full trace
and log schema.

## The scheduled health check

Two Cloud Run jobs triggered by Cloud Scheduler, running `scripts/probe.py` as a dedicated service account
(`dgem-probe-sa`, allowed only to call the Vertex endpoint and invoke the gateway and serving services).

| Job | Schedule (UTC) | Targets | What it checks |
| :--- | :--- | :--- | :--- |
| `dgem-probe-hourly` | `7 * * * *` | Vertex endpoint; gateway | Vertex: `/health`, the API contract (valid and error cases must return the expected status and answers), 20 latency requests. Gateway: `/health`, 3 decisions through the default route, which must be answered by Vertex |
| `dgem-probe-daily` | `17 6 * * *` | Cloud Run serving service | Same as Vertex; the first request wakes the scale-to-zero service, then it idles back to zero |

Each target writes one JSON log line: `probe_event="dgem.probe"`, `probe_target`, `probe_ok`, `probe_failures`,
`probe_latency_p50_ms`, `probe_latency_p95_ms`, `probe_denoise_p50_ms`, `probe_version`. Find them in Logs Explorer
with `jsonPayload.probe_event="dgem.probe"`.

Cost: the hourly job uses the always-warm endpoint (a few seconds of CPU per run); the daily job wakes one Cloud Run
GPU and keeps it up until it scales back to zero.

```bash
GCP_PROJECT=<PROJECT> \
PROBE_VERTEX_URL=https://<ENDPOINT_ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>/invoke \
PROBE_GATEWAY_URL=https://<gateway-service>.run.app \
PROBE_CLOUDRUN_URL=https://<CLOUD_RUN_URL> \
PROBE_GATEWAY_AUDIENCE=<IAP programmatic client ID, if the gateway uses IAP> \
./scripts/deploy_probe.sh
gcloud run jobs execute dgem-probe-hourly --region=<REGION> --wait     # run once now
```

- Schedules and options: `HOURLY_SCHEDULE`, `DAILY_SCHEDULE`, `SCHEDULE_TZ`, `PROBE_LATENCY_REQUESTS`,
  `PROBE_DAILY=off` (skip the Cloud Run job); re-run the script to apply. To pause the health check,
  `gcloud scheduler jobs pause dgem-probe-hourly --location=<REGION>`.
- **Gateway with IAP:** IAP accepts a service account's ID token only if its audience is one of the gateway's
  programmatic OAuth clients (`gcloud iap settings get --resource-type=cloud-run --service=<gateway> --region=<REGION>`,
  `programmaticClients`), and the account needs the IAP-secured Web App User role; the script grants it when
  `PROBE_GATEWAY_AUDIENCE` is set. Use the gateway's `run.app` URL, not a custom domain.
- **After changing the Vertex endpoint** (new endpoint ID), re-run the script with the new values.

## Alerts: recommended policies and what to do

The repository does not create alert policies for you. These are the policies we recommend; create them in Cloud
Monitoring (console, `gcloud` or Terraform) on the signals named in the second column, with a notification channel
such as `<EMAIL>`, and link each one back to this page. The thresholds are starting points: tune them to your own
traffic and baseline.

| Alert | Signal and suggested condition | Likely causes | What to do |
| :--- | :--- | :--- | :--- |
| **Vertex endpoint has no available replica** (critical) | Vertex AI replica count for the endpoint < 1, or the metric missing, for 5 min | Replica crashed or is restarting; endpoint scaled to 0; model undeployed | Check `availableReplicaCount` (`gcloud ai endpoints describe <ENDPOINT_ID> --region=<REGION>`). A `vertex_first` gateway serves from Cloud Run meanwhile. If it doesn't recover in ~15 min, redeploy or roll back ([runbook](/dgem/operate/runbook/#roll-back)) |
| **Vertex 5xx/429 error rate** | Vertex AI response codes: 5xx + 429 > 1% of responses over 10 min | Overload (`503 server busy`: more than `MAX_INFLIGHT` queued), replica restart, bad image | Check load and replica count; raise max replicas or `MAX_INFLIGHT`; if it started with a new image, roll back |
| **Gateway failing over from Vertex to Cloud Run** | Log-based metric on gateway decisions: share with `dgem_backend_requested="vertex_first"` and `dgem_backend="cloudrun"` > 20% over 30 min (at least 5 decisions) | Vertex unhealthy or updating | Usually accompanies one of the Vertex alerts; users see slower answers (and a cold start if Cloud Run was idle). Fix Vertex |
| **Gateway decision latency p95** (warning) | `dgem_total_wall_ms` p95 for Vertex-answered decisions above a threshold you set from your normal p95 (for example 2,000 ms) over 15 min | Queueing under load, very long prompts or many-question policies, reasoning (`think`) templates | Compare with your normal latency. If it's load, add replicas; if it's one template, it may just be heavy |
| **Scheduled health check failed** | Log match on `jsonPayload.probe_event="dgem.probe" AND jsonPayload.probe_ok=false` in the last ~70 min | API contract changed (new image), endpoint down, auth or IAM change for the probe account, gateway fell back to Cloud Run | Read `jsonPayload.probe_failures` on the failing log line; it names the failing case or request |
| **Scheduled health check not running** (warning) | No `dgem.probe` log line for the Vertex target for 2 h | Scheduler paused or deleted, job failing to start, image pull or IAM problem | `gcloud scheduler jobs describe dgem-probe-hourly --location=<REGION>`; run the job by hand and read its logs |
| **Health-check latency to Vertex** (warning) | `probe_latency_p50_ms` above your normal by a margin you choose | Slower serving image, GPU contention, network | Check `probe_denoise_p50_ms` (GPU time): if GPU time rose, it's the model server; if only end-to-end rose, it's network or queueing |

If you run the [regression matrix](/dgem/operate/regression-matrix/) as a scheduled job (`--log-json` prints one JSON event line
per step, `--upload` copies the run directory to Cloud Storage), you can alert on a FAIL verdict in those logs too.

## Removing the monitoring

Delete your alert policies, the scheduler jobs and Cloud Run jobs (`dgem-probe-hourly`, `dgem-probe-daily`), the
`dgem-probe-sa` account, the log-based metrics listed above and the dashboard.

## Manual checks

The health check runs a subset of the manual tools used when promoting a serving image
([runbook](/dgem/operate/runbook/#promote-a-new-serving-image)): `scripts/contract_diff.py`, `scripts/serving_speed.py` and
`scripts/coldstart_probe.py`. Run those by hand before and after changes; the scheduled check catches regressions
in between.
