---
title: "To Batch or Not to Batch"
description: "How to run dgem decisions in bulk: a warm endpoint, Cloud Run Jobs, Dataflow or Vertex AI batch prediction. Measured startup, throughput and cost per 1,000 decisions from 1,000 to 1,000,000 rows, rules of thumb, platform limits, and where bulk decisions pay off."
---

# To Batch or Not to Batch

Most dgem traffic is interactive: an agent or a service asks a question and waits about 150 ms for the answer. Some
work isn't like that. A policy changes and you want to re-apply it to last year's tickets; a data team wants every
document in a bucket labelled before training; a risk team wants a backlog of alerts pre-triaged overnight. Nobody is
waiting on any single answer, but there are a lot of them.

This page is about that second kind of work: what changes when you run decisions in bulk, which Google Cloud option
to use for which job, and what it costs. Every number is measured unless it is labelled as a projection; the method
is at the end.

## The short answer

| Your situation | Use | Why |
| :--- | :--- | :--- |
| A dgem endpoint is already running and has spare capacity | **Send it the rows** ([fan-out](#fan-out-to-an-endpoint)) | No startup, no new infrastructure. If you already pay for the endpoint, the extra cost is close to zero |
| Fewer than about 10,000 rows, no endpoint running | **A scale-to-zero Cloud Run service**, then delete it or let it scale down | First decision in about 2 minutes. A job spends about 5 minutes starting, longer than the work |
| 10,000 rows to tens of millions | **Cloud Run Jobs** | Lowest cost per decision we measured ($0.011 per 1,000 short decisions), no cluster to run, scales out by tasks |
| The batch is one step of a data pipeline (BigQuery in, joins, BigQuery out) | **Dataflow** | Reads and writes your data where it lives; costs about 1.5× Cloud Run per GPU-hour, and GPU capacity can be hard to get |
| You need Vertex AI batch prediction specifically | Vertex AI batch prediction, **L4 only** | It rejects RTX PRO 6000; on L4 it was the slowest and least predictable option |

Two settings matter more than the platform:

1. **Configure the server for throughput.** The serving defaults cap requests in flight at 8 to keep latency low. A
   batch configuration (bigger KV cache, 64 in flight) decided **85–89 short decisions per second on one RTX PRO
   6000 instead of 37–42**, which halves the cost on every platform.
2. **Use a Blackwell GPU (RTX PRO 6000).** On an L4 the same decisions ran at 13–14 per second, about 6× slower,
   which made L4 runs 3–4× more expensive per decision despite the lower hourly price.

## What batch does and doesn't change

dgem answers each question in one forward pass, and the server already batches concurrent requests on the GPU
(continuous batching). So a "batch platform" doesn't make any single decision faster or cheaper. What it changes:

- **Startup.** Every batch job pays to get a GPU, pull a 10 GB image, load 17.5 GiB of weights and warm up, before
  the first decision. That cost is fixed per worker, so it dominates small jobs and disappears into large ones.
- **Throughput per GPU.** Set by the GPU type, the server configuration and input length, not by the platform. All
  four options ran the same image, model and policy and returned the same answers (see [Method](#method)).
- **Parallelism.** How many GPUs you can use at once, which sets how fast a big job finishes. In practice that is
  limited by GPU quota and capacity more than by the platform.
- **Plumbing.** Where the rows come from and where the answers go: files in Cloud Storage, a BigQuery table, an
  existing pipeline.

The cost of a job with `N` rows on `W` GPUs is therefore roughly:

```text
cost = hourly rate × W × (startup per worker + N / (W × decisions per second per GPU))
wall time = time to first decision + N / (W × decisions per second per GPU) + time to finish
```

## The options

| | Endpoint fan-out | Cloud Run Jobs | Dataflow | Vertex AI batch prediction |
| :--- | :--- | :--- | :--- | :--- |
| What runs | Your client sends rows to a running dgem service (Cloud Run or Vertex AI) | N tasks, each starts dgem inside its container and decides its share of the files | A Beam pipeline; each worker starts dgem once in `RunInference` | Vertex starts nodes with your container and sends it batches of rows |
| Input / output | Anything your client reads | Files in Cloud Storage | Anything Beam reads and writes (Cloud Storage, BigQuery, Pub/Sub, ...) | JSONL or BigQuery |
| GPU tested | RTX PRO 6000 | RTX PRO 6000 | L4 (RTX PRO 6000 was out of stock) | L4 (RTX PRO 6000 not accepted) |
| Price per GPU-hour, `us-central1` list | $3.19 (Cloud Run, 20 vCPU / 80 GiB); Vertex `g4-standard-48` $5.85 | $3.19 | $4.65 (`g4-standard-48` + RTX PRO 6000), $1.80 (`g2-standard-16` + L4) | $1.49 (`g2-standard-16` + L4) |
| Limits we hit | The client is a single point of failure | 1 hour per GPU task; GPU quota shared with your Cloud Run services | Custom model handler needed; G4 stockouts in all three zones | ~20 min to provision; no RTX PRO 6000 |
| Recipe | [fan-out](../../deploy/batch/README.md#4-fan-out-to-a-running-endpoint) | [Cloud Run Jobs](../../deploy/batch/README.md#3-cloud-run-jobs-recommended-above-10k-rows) | [Dataflow](../../deploy/batch/README.md#5-dataflow) | [Vertex batch](../../deploy/batch/README.md#6-vertex-ai-batch-prediction) |

## Measured results

The workload: re-applying a comment moderation policy (3 questions: toxic yes/no, category out of 6, severity out of
4) to distinct public news comments ([civil_comments](https://huggingface.co/datasets/google/civil_comments)), about
345 prompt tokens per decision including the policy. One region (`us-central1`), one GPU per worker unless stated.

### Time to the first decision

| Option | Time to first decision | Of which |
| :--- | :--- | :--- |
| Endpoint already warm | 3 s | — |
| Cloud Run service scaled to zero | 2.2 min | Instance start, weights over Direct VPC egress, warmup |
| Cloud Run Jobs | 4.5–5.7 min (up to 23 min when queued on GPU quota) | ~2 min to get a GPU and start the image, ~2–2.5 min to load and warm up |
| Dataflow (L4) | 8.8–8.9 min | ~5.5 min to provision the worker and stream the image, ~3 min to load |
| Vertex AI batch prediction (L4) | 21.6–23.2 min | 16–18 min before the container started, ~3.6 min to load, ~1.3 min until the first batch |

After the last decision, Dataflow and Vertex AI took another 2–3 minutes to finish writing and shut down.

### Decisions per second, per GPU

| GPU | Server configuration | Short comments (~345 tokens) | Long threads (~2,400 tokens) |
| :--- | :--- | :---: | :---: |
| RTX PRO 6000 | Serving defaults (`MAX_INFLIGHT=8`, 2 GiB KV cache) | 37–42 | — |
| RTX PRO 6000 | Batch (`MAX_INFLIGHT=64`, 12 GiB KV cache) | **85–89** | **14.5** |
| L4 | Serving defaults | 10.8 | — |
| L4 | Batch | 13.3–13.9 | — |

Throughput per GPU was the same on every platform for a given GPU and configuration (Dataflow and Vertex AI on L4:
13.6 and 13.3 per second). Input length is the other big lever: threads with 7× the tokens of a comment ran about 6× slower.

### Cost per 1,000 decisions

Measured runs, list prices, everything the run was billed for (startup included):

| Rows | Option | GPUs | Wall time | Cost | Per 1,000 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1,000 | Cloud Run service, cold, serving defaults | 1 | 2.6 min | $0.14 | $0.138 |
| 1,000 | Cloud Run Jobs, batch | 1 | 5.3 min | $0.16 | $0.162 |
| 1,000 | Dataflow, L4 | 1 | 13.4 min | $0.30 | $0.298 |
| 1,000 | Vertex AI batch, L4 | 1 | 26.9 min | $0.63 | $0.631 |
| 10,000 | Cloud Run service, warm, serving defaults | 1 | 4.1 min | $0.22 | $0.022 |
| 10,000 | Cloud Run Jobs, serving defaults | 1 | 8.9 min | $0.40 | $0.040 |
| 10,000 | Cloud Run Jobs, batch | 1 | 9.9 min | $0.23 | $0.023 |
| 10,000 | Dataflow, L4, batch | 1 | 23.7 min | $0.62 | $0.062 |
| 10,000 | Vertex AI batch, L4, batch | 1 | 40.5 min | $0.94 | $0.094 |
| 10,000 long threads | Cloud Run Jobs, batch | 1 | 30.8 min | $0.89 | $0.089 |
| 50,000 | Cloud Run service, warm, serving defaults | 1 | 22.8 min | $1.21 | $0.024 |
| 100,000 | Cloud Run Jobs, batch | 1 | 25.0 min | $1.14 | **$0.011** |
| 100,000 | Dataflow, L4, batch | 3 | 52.5 min | $4.41 | $0.044 |
| 1,000,000 | Cloud Run Jobs, batch, 6 tasks of ~170,000 rows | 2 at a time | 108 min | $11.05 | **$0.011** |

Vertex AI batch costs are an upper bound (job start to end); its billed node time isn't reported for custom
containers. The 1,000,000-row job ran two tasks at a time because the project's Cloud Run GPU quota was 3 and a
serving service held one; with more quota the same job finishes in about 35 minutes on 6 GPUs for the same cost.

### Projections

From the measured startup and throughput, for short decisions, with each worker sized to decide for under an hour
(projections, not measured runs):

| Rows | Warm endpoint, batch settings | Cloud Run Jobs (RTX PRO 6000) | Dataflow (RTX PRO 6000, if you get capacity) | Dataflow (L4) | Vertex AI batch (L4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1M | $10 | $11 (4 GPUs, 54 min) | $18 (4 GPUs, 59 min) | $43 (23 GPUs, 65 min) | $35–47 (23 GPUs, 79 min) |
| 10M | $103 | $108 (36 GPUs) | $173 (36 GPUs) | $424 (223 GPUs) | $351–470 (228 GPUs) |
| 100M | $1,029 | $1,076 (353 GPUs) | $1,729 (353 GPUs) | $4,243 (2,229 GPUs) | $3,507–4,702 (2,279 GPUs) |

The warm endpoint column is GPU time at Cloud Run rates; if the endpoint is running anyway and has spare capacity,
the extra cost is close to zero. With the serving defaults instead of the batch settings, the RTX PRO 6000 columns roughly double.
At about $0.01 per 1,000 decisions, 100 million short decisions cost about $1,000 of GPU time.

How long it takes is set by how many GPUs you can get: one RTX PRO 6000 decides about 300,000 short rows an hour in
the batch configuration, so 10 million rows take about 3.5 hours on 10 GPUs.

## Rules of thumb

1. **Use the endpoint you already pay for.** One warm Vertex AI `g4-standard-48` replica handles about 80 short
   decisions per second, roughly 7 million a day. If it is idle at night, a nightly batch through it costs nothing
   extra. Keep the client's concurrency at the endpoint's `MAX_INFLIGHT` (8 per replica by default) and add a
   little queue, or interactive users will wait behind your batch. If the endpoint requires a service key
   (`DGEM_SERVER_KEY`), run the fan-out as a service that holds the key
   ([recipe](../../deploy/batch/README.md#4-fan-out-to-a-running-endpoint)).
2. **Below ~10,000 rows, avoid jobs.** The work takes seconds to minutes and a job spends about 5 minutes (Cloud
   Run), 9 minutes (Dataflow) or 22 minutes (Vertex AI) before the first decision.
3. **Above ~10,000 rows, use a job with the batch configuration.** Cloud Run Jobs was the cheapest per decision at
   every size we measured, and the startup cost becomes noise above ~100,000 rows.
4. **Make tasks resumable.** A Cloud Run GPU task can run for at most an hour, and any long job can be preempted.
   Write one output file per input file and skip files that already exist, so a retried task resumes.
5. **Budget for the second stage.** Hesitation-gating hands the unsure rows to a stronger model or a person. In this
   run 10% of the rows had more than 50% hesitation on the toxicity question, and they were where the errors were:
   against the dataset's labels, dgem agreed on 89% of the confident rows and 51% of the hesitant ones (100,000
   rows). Sending the hesitant rows to an LLM can cost more than dgem's pass over all the rows, so run it as a second
   batch over just those rows.
6. **Check quota before the run.** Batch jobs draw on the same GPU quota as your scale-to-zero services. A batch
   that takes every Cloud Run GPU in the project can leave your failover service unable to start. Use a separate
   project or ask for enough quota for both.

## Platform notes

**Cloud Run Jobs.** A GPU task is limited to 1 hour, so split the input into parts and choose `--tasks` so each task
gets at most about 250,000 short rows. `--parallelism` above your Cloud Run GPU quota is rejected at deploy time;
executions that ask for GPUs other jobs or services are using wait, which showed up as 13–20 minute starts. Updating a
job while an execution is running changes that execution's retries: use one job per run.

**Dataflow.** Beam's built-in vLLM handler starts stock vLLM and sends free-form prompts, so it can't serve dgem. The
recipe's `ModelHandler` starts the dgem image's own server once per worker and sends each batch to it concurrently.
Run one SDK process per worker (`--experiments=no_use_multiple_sdk_containers`) so the 26B model loads once per GPU.
Install Beam in a virtual environment *with pip* (the SDK harness checks for it). `g4-standard-48` workers failed with
`ZONE_RESOURCE_POOL_EXHAUSTED` in every `us-central1` zone during our tests, and flex-start provisioning found no
capacity in two one-hour windows; L4 workers started in about 5 minutes. A Compute Engine reservation for the same
machine (which Dataflow consumes automatically when the worker machine type and GPU match) also failed: 108 attempts
over two hours in every zone where the project had RTX PRO 6000 quota, all out of capacity. If you need RTX PRO 6000
on Dataflow, secure the capacity before you plan the job (a reservation made well ahead, or a future reservation),
and check your `GPUS_PER_GPU_FAMILY` quota per region: it was 0 in most regions for our project. Cloud Run's GPU
pool, which has its own quota, gave us RTX PRO 6000 GPUs within minutes the same day.

**Vertex AI batch prediction.** It accepted `g2-standard-16` with an L4 and rejected RTX PRO 6000 on
`g4-standard-48`. The container needs an adapter for the `{"instances": [...]}` request format
(`batch_worker.py vertex-adapter`). It took 16–18 minutes to start the container and sent only a few concurrent
batches, so even 1,000 rows took 27 minutes.

**Reading JSONL in Python.** Split on `"\n"`, not `str.splitlines()`: user text contains U+2028 and other characters
that `splitlines()` treats as line breaks, which cuts rows in half.

## Where bulk decisions pay off

Bulk decisions make sense where you have a large pile of existing items, a policy you can write down as a few
questions with fixed options, and a reason to look at all of them again. dgem's joint readout of several questions
in one pass and its calibrated hesitation are what make it cheap enough to do at scale and safe enough to automate
the confident part.

- **Backtest a policy before you ship it.** Run last month's traffic through the new policy template and compare
  the decisions with the old one: how many items change category, which ones, how hesitant the model is about them.
  This is the batch version of policy-as-code.
- **Trust and safety.** Re-apply changed community guidelines to a comment or listing archive; sweep user-generated
  content for a newly prohibited category; re-check items that were approved under an older policy.
- **Support and customer experience.** Tag a ticket history with root cause and product area; migrate tickets to a
  new routing taxonomy; score chat transcripts against a QA rubric; code open-text survey answers into themes.
- **Security operations.** Pre-triage an alert backlog (severity, likely false positive, owning team); sweep a
  mail archive for phishing after a new campaign is found; classify findings from a code scanner by exploitability.
- **Data and machine learning.** Filter a training corpus for quality, safety and personal data; label data for
  distillation or evaluation; check a retrieval corpus for stale, duplicate or poisoned documents; grade a large
  set of model transcripts as a judge, escalating only the hesitant ones.
- **Risk and compliance.** Pre-triage anti-money-laundering alerts; route insurance claims; assign dispute reason
  codes; flag clauses across a contract repository; classify records for retention.
- **Catalogs and content.** Normalize product categories to a new taxonomy (and discover missing categories with
  [taxonomy discovery](../policies/taxonomy-discovery.md)); check listings against marketplace rules; tag a media
  archive.

In each case the useful output is not only the label but the hesitation: the confident majority can be applied
automatically, and the hesitant minority goes to a person or a stronger model.

## Method

- **Workload.** 1,000,000 distinct comments from `google/civil_comments` (train, deduplicated, at least 20
  characters, shuffled with a fixed seed). Smaller sizes are prefixes of the same order, so 1,000 ⊂ 10,000 ⊂ 100,000
  ⊂ 1,000,000. Long inputs: 10,000 threads assembled from other comments, 2,400 prompt tokens on average (10th–90th percentile 1,600–3,200). One policy
  (moderation, 3 questions, `samples: 1`), rendered once and sent with every row.
- **Same software everywhere.** One image (the dgem v0.3.4 serving image plus the Beam SDK and the batch runner),
  same weights, same runner code for the request loop. Only the platform and the server configuration change.
- **Same answers everywhere.** On the first 2,000 rows, every option agreed with a reference run on the toxicity
  answer for 96.5–97.8% of rows (two runs of the same configuration also differ by about 2%, mostly on hesitant
  rows), with a mean difference in the yes-probability of 0.02–0.035. This is a consistency check, not an accuracy
  study.
- **Timing.** Time to first decision is from submitting the job to the first decision returned. Per-worker phases
  come from the runner's metrics and from the platforms' logs.
- **Cost.** `us-central1` list prices on 2026-10-10 from the Cloud Billing catalog (Cloud Run, Vertex AI, Compute
  Engine) and the Dataflow pricing page, no discounts. Cloud Run Jobs: sum of task durations. Dataflow: the job's
  own vCPU, memory, disk and GPU time metrics. Cloud Run service: instances × wall time, assuming the service is
  deleted or scales down right after. Vertex AI batch: job start to end (upper bound).
- **Limits of this test.** One region, one day. GPU availability (Dataflow RTX PRO 6000 stockouts, Cloud Run GPU
  quota of 3) shaped which runs were possible; the Dataflow RTX PRO 6000 and the 10M–100M rows are projections.
  Throughput was measured with up to 3 GPUs at once.
