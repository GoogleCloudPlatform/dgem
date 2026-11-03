---
title: "Step 3: Deploy on Cloud Run"
description: "The definitive guide to running DiffusionGemma on a scale-to-zero Cloud Run GPU service: prerequisites, weights, secrets, IAM, Direct VPC egress, deploy, verify, measured cold start, updates, rollback and troubleshooting."
---

# Step 3: Deploy on Cloud Run

A Cloud Run GPU service is the **walk** stage: your own endpoint that costs nothing while idle and wakes in a few
minutes. It can also be the failover tier behind a production Vertex endpoint; for whole datasets see
[To batch or not to batch](batch.md). Everything below uses
`scripts/deploy_cloudrun_vllm.sh`, which encodes the configuration we tested; placeholders are `<PROJECT>`,
`<REGION>` and `<GROUP>`.

## What you get

- One RTX PRO 6000 GPU instance (vision tower on) that scales to zero when idle: no cost while idle.
- Cold start (idle → warmed) of about 2.5 minutes in our tests with the default configuration (§7).
- Warm latency and throughput depend on your policies and concurrency; measure them with
  `scripts/serving_speed.py` ([Latency and capacity](../operate/latency-capacity.md)).

## 1. Prerequisites

- A project with billing, `gcloud` logged in, and these APIs enabled:
  `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, `storage.googleapis.com`,
  `secretmanager.googleapis.com`, `compute.googleapis.com`.
- **Cloud Run GPU quota** for RTX PRO 6000 in your region (new projects often have 0):

  ```bash
  gcloud beta quotas info describe NvidiaRtxPro6000GpuAllocNoZonalRedundancyPerProjectRegion \
    --service=run.googleapis.com --project=<PROJECT> --format=json \
    | jq -r '.dimensionsInfos[] | select(.dimensions.region == "<REGION>") | .details.value'
  ```

  Request an increase in the console (IAM & Admin → Quotas) if it is 0. If Cloud Run GPU quota is unavailable but
  Vertex AI serving quota exists, go straight to [Production on Vertex AI](vertex.md).
- A Google group (or user) that may call the service: `<GROUP>`.

## 2. Stage the weights in Cloud Storage

The service reads the 17.53 GiB NVFP4 checkpoint from a regional bucket in your project. The deploy script
stages it automatically on first run; to do it explicitly:

```bash
GCP_PROJECT=<PROJECT> ./scripts/stage_model_gcs.sh      # creates gs://dgem-weights-<PROJECT>/dgemma/ via Cloud Build
```

## 3. (Optional) Hugging Face token in Secret Manager

The public checkpoint does not need a token. If you use a gated model or hit Hugging Face rate limits, store the
token as a secret, never as a plain environment variable; the deploy script mounts `dgemma-hf-token` as `HF_TOKEN`
when it exists:

```bash
printf %s "$HF_TOKEN" | gcloud secrets create dgemma-hf-token --project=<PROJECT> --data-file=-
gcloud secrets add-iam-policy-binding dgemma-hf-token --project=<PROJECT> \
  --member=serviceAccount:dgemma-gpu-sa@<PROJECT>.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
```

Rotate by adding a new version and disabling the old one; new instances read `:latest`.

## 4. Networking: Direct VPC egress (the biggest cold-start win)

On a cold start the container copies the weights from Cloud Storage into memory with 64 parallel range requests
while vLLM initialises. Routed through a VPC subnet with **Private Google Access**, that copy was several times
faster in our tests than over the default public egress (about a minute and a half instead of six or more). The
script enables it by default on the `default` network; make sure the subnet has Private Google Access:

```bash
gcloud compute networks subnets update default --region=<REGION> --enable-private-ip-google-access
```

With `--vpc-egress=all-traffic` and no Cloud NAT the service has **no public internet access**. The serving
container only needs Cloud Storage and the metadata server; add Cloud NAT if you need the internet (for example to
download weights from Hugging Face at boot). Use another network with `CLOUDRUN_NETWORK`/`CLOUDRUN_SUBNET`, or
disable with `CLOUDRUN_VPC_EGRESS=off`.

## 5. Deploy

```bash
GCP_PROJECT=<PROJECT> GCP_REGION=<REGION> ALLOW_GROUP=<GROUP> \
CLOUDRUN_IMAGE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag> \
./scripts/deploy_cloudrun_vllm.sh
```

`make cloudrun-deploy` runs the same script. Build the lean image into your project first
(`GCP_PROJECT=<PROJECT> make cloudrun-build`, see [Build the serving image](public-images.md)) and pass the tag it
prints; without `CLOUDRUN_IMAGE` the script uses `<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:latest`.

What the script configures:

| Setting | Value | Why |
| :--- | :--- | :--- |
| GPU | 1× `nvidia-rtx-pro-6000` (`CLOUDRUN_GPU_TYPE`), 20 vCPU, 80 GiB | Native NVFP4; room for the vision tower and the in-memory weight copy |
| Scaling | min 0 (`CLOUDRUN_MIN_INSTANCES`), max 1, concurrency 32 | Scale to zero; one GPU per service |
| Identity | service account `dgemma-gpu-sa`, read-only on the weights bucket | Least privilege |
| Access | `--no-allow-unauthenticated`; invoker for `<GROUP>` and the gateway service account | Callers need an ID token |
| Weights | bucket mounted at `/mnt/gcs`, copied to memory at boot (`COPY_TO_SHM=1`) | Fast sequential load instead of random reads over FUSE |
| Network | Direct VPC egress, `default`/`default`, all traffic | Much faster weight copy (§4) |
| Serving | `DEFAULT_SAMPLES=1`, `MAX_INFLIGHT=8`, `DISABLE_MM=0` | Fast default reads; queue bursts instead of overloading the engine; images on |
| Startup | probe on `/health` for up to 10 minutes; self-warmup before serving | First real request doesn't pay kernel warmup |

The script injects the repository's `deploy/cloudrun/entrypoint.sh` at deploy time, so entrypoint changes take
effect on redeploy without rebuilding the image.

**L4 instead of RTX PRO 6000:** `CLOUDRUN_GPU_TYPE=nvidia-l4 DISABLE_MM=1` (24 GB GPU, 32 GiB RAM, no vision tower,
weights streamed from the bucket). It is slower per request.

## 6. Verify

```bash
URL=$(gcloud run services describe dgemma --project=<PROJECT> --region=<REGION> --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

curl -s -H "Authorization: Bearer $TOKEN" "$URL/health"
# during boot: {"phase": "staging_tmpfs", "bytes_staged_gb": ..., "vllm_ready": false}
# ready:       {"status": "ok", "vllm_ready": true, "phase": "ready", "warmed": true, "warmup_s": 14.6}

./bin/dgem decide -u "$URL/v1" --gcp-auth -t templates/support_triage.json.tmpl \
  -v 'ticket=Database connection pool exhausted' --stats
```

`/health` answers 200 as soon as the container is up; wait for `vllm_ready` and `warmed` before sending load.

## 7. Cold start: what to expect and your options

Approximate time from a new instance starting until it is warmed, in our tests:

| Configuration | Idle → warmed | Extra cost |
| :--- | :--- | :--- |
| **Lean image + Direct VPC egress (default)** | **about 2.5 minutes** | None |
| Lean image, public egress | about 7.5 minutes (slow weight copy) | None |
| Weights baked into the image | dominated by the image pull on a fresh instance (well over 10 minutes) | None |
| Minimum 1 instance (`CLOUDRUN_MIN_INSTANCES=1`) | 0 | One GPU billed continuously |
| Vertex AI endpoint (min replicas ≥ 1) | 0 | Per replica-hour |

- Besides the weight copy, a cold start includes vLLM start-up (overlapped with the copy), loading weights onto the
  GPU and building caches, and a short self-warmup.
- The baked-weights image does not help on Cloud Run: every new image or revision can pay the long pull.
  Keep it for offline hosts.
- To hide cold starts from users, put a [gateway](gateway.md) in front that routes to an always-warm Vertex endpoint
  first and uses this service for failover and batch work.
- Measure your own cold start with `scripts/coldstart_probe.py`.

## 8. Update safely (blue/green) and roll back

```bash
# Deploy a new image as a tagged revision with no traffic
gcloud run services update dgemma --project=<PROJECT> --region=<REGION> \
  --image=<NEW_IMAGE> --no-traffic --tag=candidate

# Compare it with the serving revision, then switch
python3 scripts/contract_diff.py --target old=$URL --target new=https://candidate---<SERVICE_HOST>
gcloud run services update-traffic dgemma --project=<PROJECT> --region=<REGION> --to-tags=candidate=100

# Roll back
gcloud run services update-traffic dgemma --project=<PROJECT> --region=<REGION> --to-revisions=<PREVIOUS_REVISION>=100
```

For a new serving image, run the full comparison first: [Promote a new serving image](../operate/runbook.md#promote-a-new-serving-image).

## 9. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `429` / connection refused right after deploy or idle | Instance still loading (vLLM not ready) | Poll `/health` until `vllm_ready`; retry with backoff (`--http-retries 3`) |
| Container restarts ~10 minutes into boot | Out of memory (vision tower + in-memory weights on a 32 GiB instance) | Use RTX PRO 6000 (80 GiB), or `DISABLE_MM=1` on L4 |
| Weight copy takes 6+ minutes | Public egress | Enable Direct VPC egress with Private Google Access (§4) |
| Weights download from Hugging Face fails | No internet with all-traffic VPC egress and no NAT | Stage weights in Cloud Storage (§2) or add Cloud NAT |
| `503 server busy` | More than `MAX_INFLIGHT` decisions queued for 30 s | Reduce client concurrency, raise `MAX_INFLIGHT`, or add capacity (Vertex replicas) |
| `403` from the service | Caller lacks `run.invoker` or sends no ID token | Add the caller to `<GROUP>`; use `--gcp-auth` |
| Deploy fails with GPU quota error | No Cloud Run GPU quota in the region | Request quota, or use Vertex AI |

## 10. Tear down

```bash
gcloud run services delete dgemma --project=<PROJECT> --region=<REGION> --quiet
```

An idle service already costs nothing; delete it when you no longer need the URL. The weights bucket bills for
storage (~18 GB) until deleted.

## Next

[Production on Vertex AI](vertex.md) for zero cold start, or [Gateway and routing](gateway.md) to put one URL in
front of both.
