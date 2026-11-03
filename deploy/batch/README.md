# dgem batch recipes

Run one policy over a whole dataset (JSONL in Cloud Storage) on Cloud Run Jobs, Dataflow or Vertex AI batch
prediction, or fan it out to a running dgem endpoint. When to use which: [docs/deploy/batch.md](../../docs/deploy/batch.md).

| File | What it is |
| :--- | :--- |
| `batch_worker.py` | Standard-library runner. `job`: starts the dgem server inside the container and decides this task's share of the input files. `remote`: the same loop against a running endpoint. `vertex-adapter`: serves Vertex batch prediction's `{"instances": [...]}` requests. |
| `dgem_beam.py` | Apache Beam pipeline with a custom `ModelHandler` for `RunInference` (Beam's built-in vLLM handler can't serve dgem). |
| `Dockerfile`, `cloudbuild.yaml` | The batch image: the dgem serving image plus the Beam SDK worker and the two scripts. |
| `moderation_backfill.json.tmpl` | Example policy: 3 questions about a user comment. |

Placeholders: `<PROJECT>`, `<REGION>`, `<BUCKET>` (a `gs://` bucket you own), `<TAG>`, `<SA>` (a service account
with read access to the weights bucket and the image, and write access to `<BUCKET>`; add `roles/dataflow.worker`
for Dataflow).

## 1. Input, schema and image

Input rows are JSON objects, one per line: an `id`, the field your policy's `state` reads, and anything else you want
copied to the output (`gold`, for example). Split large inputs into parts of about 10,000 rows so workers can share
them out by file.

```json
{"id": "c-0000001", "comment": "Thanks for posting this, very helpful."}
```

Render the policy once; every row reuses the schema:

```bash
dgem template render -t deploy/batch/moderation_backfill.json.tmpl \
  | awk '/^{/{p=1} /^─/{p=0} p' | jq '.schema | .samples |= tonumber' > schema.json
gcloud storage cp schema.json <BUCKET>/schemas/moderation_backfill.json
```

Build the batch image on top of the serving image you deploy today:

```bash
gcloud builds submit deploy/batch --config deploy/batch/cloudbuild.yaml --region <REGION> \
  --substitutions=_BASE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgemma:<TAG>,_IMAGE=<REGION>-docker.pkg.dev/<PROJECT>/dgem/dgem-batch:<TAG>
```

## 2. Server settings for batch

These settings were measured to double throughput per GPU against the serving defaults (85–89 against 37–42 short
decisions/s on one RTX PRO 6000). Latency per request goes up, which a batch doesn't notice:

```text
KV_CACHE_GB=12 GPU_UTIL=0.85 MAX_SEQS=64 MAX_INFLIGHT=64 DISABLE_MM=1 MAX_MODEL_LEN=8192 CANVAS=128 DEFAULT_SAMPLES=1
```

`DISABLE_MM=1` turns the vision tower off. Leave it at `0` if your policy reads images.

## 3. Cloud Run Jobs (recommended above ~10k rows)

```bash
gcloud beta run jobs create dgem-batch --project <PROJECT> --region <REGION> \
  --image <REGION>-docker.pkg.dev/<PROJECT>/dgem/dgem-batch:<TAG> \
  --command python3 --args /opt/dgem-batch/batch_worker.py,job \
  --service-account <SA> --cpu 20 --memory 80Gi \
  --gpu 1 --gpu-type nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy \
  --add-volume=name=weights,type=cloud-storage,bucket=dgem-weights-<PROJECT>,readonly=true,mount-options=enable-buffered-read=true \
  --add-volume-mount=volume=weights,mount-path=/mnt/gcs \
  --network=default --subnet=default --vpc-egress=all-traffic \
  --tasks 4 --parallelism 4 --max-retries 3 --task-timeout 1h \
  --set-env-vars=MODEL=/mnt/gcs/dgemma,COPY_TO_SHM=1,KV_CACHE_GB=12,GPU_UTIL=0.85,MAX_SEQS=64,MAX_INFLIGHT=64,DISABLE_MM=1,MAX_MODEL_LEN=8192,CANVAS=128,DEFAULT_SAMPLES=1,BATCH_INPUT=<BUCKET>/in/,BATCH_OUTPUT=<BUCKET>/out/run1/,BATCH_SCHEMA=<BUCKET>/schemas/moderation_backfill.json,BATCH_STATE_KEY=comment,BATCH_CONCURRENCY=96
gcloud run jobs execute dgem-batch --project <PROJECT> --region <REGION>
```

- A GPU task may run for at most 1 hour. Size `--tasks` so each task gets at most about 250,000 short rows
  (fewer for long inputs). The runner skips files that already have output, so a retried task resumes.
- `--parallelism` is limited by your Cloud Run GPU quota, which your serving service shares. Running batch jobs in
  the same project as a scale-to-zero failover service can take the GPU the failover needs.
- Per-task metrics (boot, server ready, first decision, end) go to `<BUCKET>/out/run1/_metrics/`.

## 4. Fan out to a running endpoint

If a dgem endpoint is already up and has spare capacity, the cheapest batch is to send it the rows:

```bash
python3 deploy/batch/batch_worker.py remote --url https://<CLOUD_RUN_URL> \
  --input <BUCKET>/in/ --output <BUCKET>/out/run2/ --schema <BUCKET>/schemas/moderation_backfill.json \
  --state-key comment --concurrency 32
```

`remote` mints an identity token from Application Default Credentials. Keep concurrency at or below what the
endpoint is configured for (`MAX_INFLIGHT` × instances, plus a little queue).

If the serving image runs with `DGEM_SERVER_KEY` set, it rejects any POST without the matching `X-DGem-Key` header.
`remote` sends that header when `DGEM_SERVER_KEY` is set in its own environment (the first entry of a
comma-separated rotation list). The key is for services only: run the fan-out as a service (a Cloud Run job or a
VM with the key from Secret Manager), never from a personal machine with the key pasted in, and never write it to a
file. People should go through your gateway instead. The `job` and `vertex-adapter` modes start their own server
without a key and remove `DGEM_SERVER_KEY` from its environment.

## 5. Dataflow

Use Dataflow when the batch is one step of a pipeline (BigQuery in, joins, BigQuery out). The launching
environment needs Python 3.12 and `apache-beam[gcp]==2.69.0`, the same versions as the image.

```bash
PYTHONPATH=deploy/batch python3 deploy/batch/dgem_beam.py \
  --input '<BUCKET>/in/part-*.jsonl' --output <BUCKET>/out/run3/part \
  --schema <BUCKET>/schemas/moderation_backfill.json --state_key comment --concurrency 64 \
  --server_env '{"DGEM_WEIGHTS_URI": "gs://dgem-weights-<PROJECT>/dgemma", "KV_CACHE_GB": "12", "GPU_UTIL": "0.85", "MAX_SEQS": "64", "MAX_INFLIGHT": "64", "DISABLE_MM": "1", "MAX_MODEL_LEN": "8192", "CANVAS": "128"}' \
  --runner DataflowRunner --project <PROJECT> --region <REGION> \
  --temp_location <BUCKET>/dataflow/temp --staging_location <BUCKET>/dataflow/staging \
  --sdk_container_image <REGION>-docker.pkg.dev/<PROJECT>/dgem/dgem-batch:<TAG> --sdk_location container \
  --machine_type g4-standard-48 \
  --dataflow_service_options 'worker_accelerator=type:nvidia-rtx-pro-6000;count:1;install-nvidia-driver:latest' \
  --experiments no_use_multiple_sdk_containers --number_of_worker_harness_threads 4 --disk_size_gb 200 \
  --num_workers 4 --autoscaling_algorithm NONE --service_account_email <SA> \
  --network default --subnetwork regions/<REGION>/subnetworks/default --no_use_public_ips
```

- `no_use_multiple_sdk_containers` keeps one SDK process per worker, so the model loads once per GPU.
- For an L4 worker use `--machine_type g2-standard-16`, `type:nvidia-l4`, and `"CANVAS": "32", "KV_CACHE_GB": "3",
  "GPU_UTIL": "0.92", "MAX_SEQS": "48", "MAX_INFLIGHT": "32"`.
- If workers don't start (`ZONE_RESOURCE_POOL_EXHAUSTED`), add `;provisioning_model:FLEX_START` to the accelerator
  option: the job waits up to an hour for capacity, using preemptible GPU quota.
- For guaranteed capacity, create a Compute Engine reservation for the exact worker shape and pin the zone:

  ```bash
  gcloud compute reservations create dgem-df --project <PROJECT> --zone <ZONE> --vm-count 4 \
    --machine-type g4-standard-48 --accelerator count=1,type=nvidia-rtx-pro-6000
  # then add to the pipeline:  --worker_zone <ZONE>
  ```

  Dataflow consumes it automatically when machine type and GPU match, and bills Dataflow prices while the job runs;
  the reservation is billed at Compute Engine prices whenever it is idle, so delete it after the job. Creating it
  needs capacity too.

## 6. Vertex AI batch prediction

Vertex batch prediction runs the same image with `batch_worker.py vertex-adapter` as the container command. At the
time of writing it accepts L4 (`g2-standard-16` + `NVIDIA_L4`) but rejects RTX PRO 6000 on `g4` machines, so it
runs about 6× slower per GPU than the other options. Upload a model with:

- container command `python3`, args `/opt/dgem-batch/batch_worker.py vertex-adapter`, predict route `/predict`,
  health route `/health`, port 8080;
- environment: the L4 settings above plus `DGEM_WEIGHTS_URI`, `BATCH_SCHEMA`, `BATCH_STATE_KEY`,
  `BATCH_CONCURRENCY=64`;

then create a batch prediction job with JSONL input and output, `manualBatchTuningParameters.batchSize` 64 and the
service account `<SA>`.
