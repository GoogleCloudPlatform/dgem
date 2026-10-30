---
title: "Operations Runbook"
description: "Operating dgem in production: health signals, routine checks, promoting a new serving image with side-by-side validation, rollback, teardown, alerting and incident playbooks."
---

# Operations Runbook

For SREs running the Vertex endpoint, the Cloud Run GPU service and the gateway described in
[From Laptop to Production](../deploy/index.md). Placeholders: `<PROJECT>`, `<REGION>`, `<ENDPOINT_ID>`,
`<your-dgem-gateway>`.

## Health signals

`GET /health` on the serving container (`/invoke/health` on Vertex) returns HTTP 200 once the container is up, with
readiness details:

| Field | Meaning |
| :--- | :--- |
| `phase` | `mounting_gcs` → `staging_tmpfs` → `loading_vllm_siglip` → `ready` |
| `bytes_staged_gb` | Weight copy progress (17.53 when done) |
| `vllm_ready` | The engine answers; decisions will succeed |
| `warmed`, `warmup_s` | The self-warmup (1-sample, 4-sample and image decisions) finished |

The gateway exposes `GET /health` (open; liveness and version only), `GET /api/status?backend=vertex|cloudrun|local`
and `GET /api/backend-config` (live Vertex state: `deployed`, replica count, machine type), and the MCP tool
`get_health_and_gpu_status`. Each backend's readiness is tracked separately, so a Vertex probe never marks Cloud Run
as warm.

## Routine checks

- Vertex: `availableReplicaCount ≥ 1` on the deployed model (`gcloud ai endpoints describe <ENDPOINT_ID>`), and a
  periodic `/invoke/health` (the gateway sends one every 4 minutes; keep the gateway at min 1 instance).
- Cloud Run GPU: scales to zero by design; verify it can wake with an occasional decision.
- Gateway: one decision per backend with `X-DGem-Backend` and check `X-DGem-Backend-Used`.
- Costs: only intended Vertex deployed models and GCE VMs exist (`gcloud ai endpoints list`,
  `gcloud compute instances list`).

## Promote a new serving image

A serving image bundles the vLLM runtime and the structured server; small changes can alter answers or break the
API. Validate every new image **next to your current deployment, in the same session**, before moving traffic:
single-image testing misses regressions that a side-by-side comparison catches.

1. **Build** into your own Artifact Registry with an immutable tag:
   `GCP_PROJECT=<PROJECT> IMAGE_TAG=$(git rev-parse --short HEAD) make cloudrun-build`
   ([Build the serving image](../deploy/public-images.md)).
2. **Deploy the candidate blue/green, without traffic:**
   - **Vertex AI:** deploy the new image next to the old model on the same endpoint at 0% traffic
     (`VERTEX_NEW_TRAFFIC=0 IMAGE_URI=<NEW_IMAGE> VERTEX_MODEL_NAME=dgemma-g4-<TAG> ./scripts/deploy_vertex_endpoint.sh`,
     [steps](../deploy/vertex.md#5-swap-images-with-zero-downtime)), or on a separate canary endpoint
     (`VERTEX_ENDPOINT_NAME=dgemma-canary-<TAG> VERTEX_MAX_REPLICAS=1`) if you want to call it directly by URL.
   - **Cloud Run:** a tagged revision with no traffic:
     `gcloud run services update dgemma --image=<NEW_IMAGE> --no-traffic --tag=candidate`; it is reachable at
     `https://candidate---<SERVICE_HOST>`.
3. **Compare** old and new with the [regression matrix](regression-matrix.md), tier **T1**, the current deployment
   as the baseline in the same session (tier **T2** as well for a release or a vLLM change):

   ```bash
   scripts/bench_matrix.py run --tier T1 --target old=<CURRENT_URL> --target new=<CANDIDATE_URL> \
     --baseline old --label image-<TAG>
   ```

   It covers the API contract (including multilingual cases with known answers), readiness, accuracy (calibration
   and JevBench ×3 on both prompt paths, intents, the Decision Index adapter suites) judged against the noise floor
   it measures, a multilingual spot check, and latency and load (modes, sweep at 16 and 32 workers, 0 errors).
   Promote only when every gate is PASS or each REVIEW is explained; a FAIL blocks. On Cloud Run also run
   `scripts/coldstart_probe.py` for both images (the matrix does not create revisions), and
   `scripts/contract_diff.py --target old=<CURRENT_URL> --target new=<CANDIDATE_URL>` for a quick contract check.
4. **Switch** without downtime: Vertex with a traffic-split update
   ([steps](../deploy/vertex.md#5-swap-images-with-zero-downtime)); Cloud Run with
   `gcloud run services update-traffic dgemma --to-tags=candidate=100`. Run matrix **T0** against the serving URL
   afterwards.
5. **Clean up:** delete canaries; after a soak period, undeploy the old Vertex model (keep its model resource for
   rollback).

The gateway follows the same pattern: `GATEWAY_TAG=candidate ./scripts/deploy_cloudrun_gateway.sh` deploys a tagged
revision with no traffic ([Gateway and routing](../deploy/gateway.md#deploy-the-gateway-on-cloud-run)).

Per-item answers can shift a few percent between vLLM versions even when accuracy is unchanged: compare totals and
agreement, not individual answers, across images.

When changing `deploy/cloudrun/server/structured_server.py`, keep it as vLLM's upstream example plus
`dgem.patch`: edit the file, then run `scripts/regen_serving_patch.sh`, which regenerates the patch against
`structured_server.upstream.py` and checks it re-applies. The patch must keep the `/health` readiness fields, HTTP/1.1
keep-alive with backlog 256, `MAX_INFLIGHT`, `DEFAULT_SAMPLES` and the `/predict` / `/rawPredict` aliases.

## Versions

Serving images, the gateway and the CLI share one version (`vMAJOR.MINOR.PATCH`, see
[CHANGELOG.md](../../CHANGELOG.md)). A running service reports it: `/health` returns `version` and `revision`,
images carry `org.opencontainers.image.version`, and `dgem --version` prints it. Builds between release tags report
`git describe` versions such as `v0.1.0-3-gabc1234`. Deploy your own images by tag or digest and record the version
in your deployment log.

## Roll back

| Component | How | Time |
| :--- | :--- | :--- |
| Vertex (old model still deployed) | Move the traffic split back | Seconds |
| Vertex (old model undeployed) | Redeploy the old model resource with `VERTEX_NEW_TRAFFIC=0`, then move traffic | ~15 min |
| Cloud Run GPU service / gateway | `gcloud run services update-traffic <SERVICE> --to-revisions=<PREVIOUS>=100` | Seconds (GPU service then cold-starts) |

## Tear down

- Vertex: `GCP_PROJECT=<PROJECT> make vertex-teardown` (or undeploy the model to keep the endpoint and its URL).
- Cloud Run: idle services cost nothing; `gcloud run services delete <SERVICE>` removes them.
- GCE VMs: `make gce-teardown` immediately after use.

## Alerting

Log-based metrics and the dashboard (`scripts/setup_cloud_monitoring.sh`), the scheduled health check, recommended
alert policies and what to do when each fires: [Monitoring and alerts](monitoring.md).

## Incident playbooks

| Symptom | Likely cause | Action |
| :--- | :--- | :--- |
| Vertex `429 ... scale-up from zero` / "not yet ready" | Replica scaled to 0 or still starting | Check `availableReplicaCount`; the gateway fails over to Cloud Run meanwhile; confirm min replicas ≥ 1 and keepalive running |
| Many `503 server busy` | Load above `MAX_INFLIGHT` for 30 s | Add replicas / raise max replicas; ask clients to back off |
| `503 Model server early terminated the request` | Serving image without HTTP keep-alive | Upgrade to an image with the keep-alive fix |
| All traffic on Cloud Run with multi-minute latencies | Vertex down and Cloud Run cold | Restore Vertex; Cloud Run warms in a few minutes |
| `/health` stuck in `staging_tmpfs` | Slow or blocked Cloud Storage access | Check Direct VPC egress and Private Google Access ([Cloud Run §4](../deploy/cloud-run.md#4-networking-direct-vpc-egress-the-biggest-cold-start-win)) |
| Replica restarts during boot | Out of host memory | Larger machine / RTX PRO 6000 instance, or `DISABLE_MM=1` |
| IAP 401/403 for a user | Not in the allowed group | Add to `<GROUP>` |
