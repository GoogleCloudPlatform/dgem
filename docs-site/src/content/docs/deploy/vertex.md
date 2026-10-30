---
title: "Step 4: Production on Vertex AI"
description: "Run DiffusionGemma on an always-warm Vertex AI dedicated endpoint (g4-standard-48 + RTX PRO 6000): deploy, verify, scale, swap images with zero downtime, and tear down."
---

A Vertex AI **dedicated endpoint** is the **run** stage: always warm, autoscaling replicas and Cloud Monitoring, on
the same container image as Cloud Run. The container is deployed with arbitrary custom routes
(`invokeRoutePrefix: "/*"`), so every server route is available under `/invoke/*`
([why](/dgem/reference/vertex-vs-cloud-run/)). Deploy it with a [service key](#1-prerequisites) so that only your
own services can make decisions on it, and give people access through the [gateway](/dgem/deploy/gateway/).

## What you get

- One `g4-standard-48` + 1× RTX PRO 6000 replica per deployed model, with the vision tower on.
- No cold start while minimum replicas ≥ 1. Deploying a new model takes roughly 10–15 minutes.
- Latency and throughput depend on your policies (questions, samples, images, reasoning budget) and concurrency;
  measure your own deployment with `scripts/serving_speed.py` (see
  [Latency and capacity](/dgem/operate/latency-capacity/)).

## 1. Prerequisites

- APIs: `aiplatform.googleapis.com`, `artifactregistry.googleapis.com`, `storage.googleapis.com`.
- **Vertex AI serving quota** for RTX PRO 6000 in your region:

  ```bash
  gcloud beta quotas info describe CustomModelServingRTXPRO6000GPUsPerProjectPerRegion \
    --service=aiplatform.googleapis.com --project=<PROJECT> --format=json \
    | jq -r '.dimensionsInfos[] | select(.dimensions.region == "<REGION>") | .details.value'
  ```

- Weights in `gs://dgem-weights-<PROJECT>/dgemma/` (see [Cloud Run §2](/dgem/deploy/cloud-run/#2-stage-the-weights-in-cloud-storage));
  the endpoint mounts them as the model artifact.
- A service account for the endpoint (default `dgemma-gpu-sa@<PROJECT>.iam.gserviceaccount.com`) with read access
  to the bucket.
- Vertex AI requires an Artifact Registry or Container Registry image. [Build the serving image](/dgem/deploy/public-images/)
  into your project (`GCP_PROJECT=<PROJECT> make cloudrun-build`) and pin the commit tag or digest it prints.
- A **service key** for the serving image, stored in Secret Manager as `dgem-server-key`. When the secret exists, the
  deploy script sets it as `DGEM_SERVER_KEY` on the model and the serving image refuses every `POST` without a
  matching `X-DGem-Key` header (`GET /health` stays open). Without the secret the script prints a warning and the
  serving image accepts any `POST` that reaches it. Give the key to the services that call the endpoint (your
  [gateway](/dgem/deploy/gateway/), batch jobs, probes), not to people:

  ```bash
  openssl rand -base64 32 | tr -d '\n' | gcloud secrets create dgem-server-key --project=<PROJECT> --data-file=-
  ```

  Vertex AI model environment variables are plain values, so anyone who can read the model resource
  (`aiplatform.models.get`) can read the key: keep that permission to the operators who deploy the endpoint.

## 2. Deploy

```bash
GCP_PROJECT=<PROJECT> GCP_PROJECT_NUMBER=<PROJECT_NUMBER> GCP_REGION=<REGION> \
IMAGE_URI=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<tag> \
./scripts/deploy_vertex_endpoint.sh
```

The default profile (`VERTEX_PROFILE=g4-rtxpro6000`) configures:

| Setting | Value |
| :--- | :--- |
| Machine | `g4-standard-48` + 1× `NVIDIA_RTX_PRO_6000` |
| Replicas | min 1 (`VERTEX_MIN_REPLICAS`), max 2 (`VERTEX_MAX_REPLICAS`), autoscale at 70% GPU duty cycle |
| Engine | `KV_CACHE_GB=12`, `MAX_SEQS=32`, `MAX_INFLIGHT=8`, `DEFAULT_SAMPLES=1`, vision on |
| Endpoint | dedicated DNS, display name `dgemma-dedicated-g4` (`VERTEX_ENDPOINT_NAME`; reused if it exists) |
| Startup | long startup probe (weights load + self-warmup before the replica takes traffic) |

The script prints the endpoint ID and the invoke URLs:

```text
https://<ENDPOINT_ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>/invoke/...
  /v1/chat/completions      structured decisions (what dgem uses)
  /v1/systemone             /v1/systemone decision protocol
  /health                   readiness
```

## 3. Verify

```bash
INVOKE=https://<ENDPOINT_ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>/invoke
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" $INVOKE/health
#   -> {"status": "ok", "vllm_ready": true, "phase": "ready", "warmed": true, ...}

# Without the service key, decision requests are refused
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" -d '{}' $INVOKE/v1/chat/completions
#   -> 401

# As an operator, with the key from Secret Manager (callers send DGEM_SERVER_KEY as X-DGem-Key)
export GCP_PROJECT=<PROJECT> GCP_PROJECT_NUMBER=<PROJECT_NUMBER>
export DGEM_SERVER_KEY="$(gcloud secrets versions access latest --secret=dgem-server-key --project=<PROJECT>)"
./bin/dgem decide --vertex-url <ENDPOINT_ID> --gcp-auth -t templates/support_triage.json.tmpl \
  -v ticket="Charged twice" --stats
./bin/dgem decide --vertex-url <ENDPOINT_ID> --gcp-auth -t templates/multimodal/bbox_localization.json.tmpl \
  -I fixtures/bbox/bbox-t1-03-offgrid-card.png -v target=checkout_summary_card
```

## 4. Size it

- Measure one replica's throughput for your policies with a concurrency sweep; past the point where requests
  start to queue (`MAX_INFLIGHT`), latency rises while throughput plateaus. Set `VERTEX_MAX_REPLICAS` for your peak;
  the endpoint adds replicas on GPU duty cycle. Details: [Latency and capacity](/dgem/operate/latency-capacity/).
- Keep minimum replicas ≥ 1 and a periodic `/invoke/health` probe (the [gateway](/dgem/deploy/gateway/) runs one every 4
  minutes). If an endpoint reports 0 available replicas it returns `429` until it scales back up.

## 5. Swap images with zero downtime

Deploy the new image next to the current one at 0% traffic, verify it on the same endpoint, then move traffic:

```bash
# 1. New model on the same endpoint, 0% traffic (the current split is kept)
VERTEX_NEW_TRAFFIC=0 GCP_PROJECT=<PROJECT> GCP_PROJECT_NUMBER=<PROJECT_NUMBER> \
IMAGE_URI=<NEW_IMAGE> VERTEX_MODEL_NAME=dgemma-g4-<TAG> ./scripts/deploy_vertex_endpoint.sh

# 2. Move traffic (deployed model IDs from `gcloud ai endpoints describe <ENDPOINT_ID> --region=<REGION>`)
curl -X PATCH -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" \
  "https://<REGION>-aiplatform.googleapis.com/v1/projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>?updateMask=trafficSplit" \
  -d '{"trafficSplit": {"<NEW_DEPLOYED_MODEL_ID>": 100, "<OLD_DEPLOYED_MODEL_ID>": 0}}'

# 3. After a soak period, undeploy the old model (its model resource stays uploaded for rollback)
gcloud ai endpoints undeploy-model <ENDPOINT_ID> --region=<REGION> --deployed-model-id=<OLD_DEPLOYED_MODEL_ID>
```

Rollback while both are deployed is instant (move the split back); after undeploying, redeploy the old model
resource (~15 minutes). Before step 2 on a new serving image, run the comparison in
[Promote a new serving image](/dgem/operate/runbook/#promote-a-new-serving-image).

## 6. Tear down

```bash
GCP_PROJECT=<PROJECT> make vertex-teardown      # undeploys models and deletes the dgemma-dedicated-g4 endpoint
```

A deployed replica bills per hour whether or not it serves traffic; tear down endpoints you don't need, and keep a
scale-to-zero [Cloud Run service](/dgem/deploy/cloud-run/) for occasional use.

## Next

[Gateway and routing](/dgem/deploy/gateway/): one URL in front of Vertex and Cloud Run, with automatic failover.
