---
title: "From Laptop to Production"
description: "The dgem developer journey: run locally, deploy a scale-to-zero Cloud Run GPU, promote to an always-warm Vertex AI endpoint, and operate it."
---

# From Laptop to Production

This track takes `dgem` from your laptop to a production endpoint and keeps it running. Identifiers such as
`<PROJECT>`, `<REGION>` and `<ENDPOINT_ID>` are placeholders. The cloud stages run a serving image that you
[build into your own Artifact Registry](public-images.md).

## The journey

| Step | Page | What you get |
| :--- | :--- | :--- |
| 1 | [Run on your laptop](laptop.md) | A first decision in minutes on Apple Silicon or a local NVIDIA GPU |
| 2 | [Use a remote GPU](remote-gpu.md) | Point the CLI, Studio and MCP at any hosted endpoint |
| 3 | [Deploy on Cloud Run](cloud-run.md) | Your own scale-to-zero GPU service: no cost when idle, a few minutes of cold start |
| 4 | [Production on Vertex AI](vertex.md) | An always-warm endpoint with autoscaling replicas and zero cold start |
| 5 | [Gateway and routing](gateway.md) | One URL for users, automatic failover from Vertex to Cloud Run, IAP |
| 6 | [Latency and capacity](../operate/latency-capacity.md) | Request design, engine settings and how to measure your own |
| 7 | [Operations runbook](../operate/runbook.md) | Health, promotion, rollback, teardown and troubleshooting |
| 8 | [Monitoring and alerts](../operate/monitoring.md) | Health checks, dashboards, alerts and what to do when they fire |

Deciding a whole dataset rather than one request at a time? [To batch or not to batch](batch.md) compares a warm
endpoint, Cloud Run Jobs, Dataflow and Vertex AI batch prediction, with measured cost per 1,000 decisions.

Hosting it yourself? [Build the serving image](public-images.md) first; [Evaluate dgem on your own GPU](evaluate.md)
covers GPU requirements, checking the install with the regression matrix, and a custom evaluation on your own
labelled data.

## Crawl → walk → run

| Stage | Platform | Use it for | Cost model | Cold start |
| :--- | :--- | :--- | :--- | :--- |
| **Crawl** | Apple Silicon (`diffgemma`, Metal, q4) or a local NVIDIA GPU | Development, writing policies, offline and private data | Your machine | None (Metal is a different engine; its latency doesn't transfer to GPU serving) |
| **Walk** | Cloud Run GPU, 1× RTX PRO 6000 | Batch evaluation, research, bursty internal use, failover | Per instance-second; **no cost when idle** | A few minutes from idle (lean image + Direct VPC egress) |
| **Run** | Vertex AI dedicated endpoint, `g4-standard-48` + 1× RTX PRO 6000 | Production: always warm, IAM, autoscaling, monitoring, images | Per replica-hour while deployed | **None** (min replicas ≥ 1) |

The same image, policies, CLI, HTTP API and MCP server work on every stage. Latency depends on your policies,
concurrency and network: measure it on your own deployment ([method](#method)).

## TL;DR recommendations

- **GPU:** use a Blackwell GPU (RTX PRO 6000). The checkpoint is NVFP4 (4-bit); Blackwell runs FP4 natively, and
  the L4 was slower on every request type in our tests. Other GPUs are untested
  ([requirements](evaluate.md#0-hardware)).
- **Production:** a Vertex AI dedicated endpoint with minimum replicas ≥ 1, autoscaling on GPU duty cycle.
- **Failover:** a scale-to-zero Cloud Run GPU service using the lean image and Direct VPC egress.
- **Bulk decisions:** Cloud Run Jobs with the batch server settings above ~10,000 rows, or your warm endpoint if it
  has spare capacity ([to batch or not to batch](batch.md)).
- **One gateway** (`dgem serve`) in front of both, routing `vertex_first` with automatic failover.
- **Requests:** `"samples": 1` by default; opt in to `4` only when you need agreement/standard error; never
  `"auto"` on latency-sensitive paths. Put all questions about one input in one request.
- **Change safely:** validate every new serving image next to your current deployment before switching traffic
  ([runbook](../operate/runbook.md#promote-a-new-serving-image)).

## Method

- Tools: [`scripts/serving_speed.py`](../../scripts/serving_speed.py) (latency and concurrency),
  [`scripts/coldstart_probe.py`](../../scripts/coldstart_probe.py) (Cloud Run cold start),
  [`scripts/contract_diff.py`](../../scripts/contract_diff.py) (API contract). Failed requests are counted as
  errors, never as zero latency.
- Use a single client in the same region as the endpoint: that measures service-side characteristics, not end-user
  internet latency.
