---
title: "Step 2: Use a Remote GPU"
description: "Point the dgem CLI, Decision Studio and MCP at a hosted DiffusionGemma endpoint: Vertex AI dedicated endpoints, Cloud Run GPU services, dgem gateways or any OpenAI-compatible host."
---

# Step 2: Use a Remote GPU

`dgem` is a thin client: every command works the same against your laptop, a Cloud Run GPU service, a Vertex AI
endpoint or a `dgem serve` gateway. If your team already runs an endpoint, this page is all you need; to run your
own, continue with [Deploy on Cloud Run](cloud-run.md).

## Choose a target

| Target | How to connect | Auth | Cold start |
| :--- | :--- | :--- | :--- |
| Vertex AI dedicated endpoint | `--vertex-url <ENDPOINT_ID>` | OAuth access token (`--gcp-auth`) | None while min replicas ≥ 1 |
| Cloud Run GPU service | `-u https://<CLOUD_RUN_URL>/v1` | ID token (`--gcp-auth`) | A few minutes after idle (scales to zero) |
| `dgem serve` gateway | `-u https://<your-dgem-gateway>/v1` | ID token / IAP (`--gcp-auth`) | Depends on the routed backend |
| Local engine | `-u http://127.0.0.1:8080/v1` | none | none |

`--stats` prints the server's GPU time and the end-to-end time of each request, so you can see what your own
network and endpoint add.

## Connect

```bash
# Vertex AI dedicated endpoint. A bare ID is expanded to the dedicated /invoke/* URL, which needs
# GCP_PROJECT and GCP_PROJECT_NUMBER (or run on GCP); otherwise pass the full invoke URL.
export GCP_PROJECT=<PROJECT> GCP_PROJECT_NUMBER=<PROJECT_NUMBER>
./bin/dgem decide --vertex-url <ENDPOINT_ID> --gcp-auth \
  -t templates/support_triage.json.tmpl -v 'ticket=Database connection pool exhausted' --stats

# Cloud Run GPU service
./bin/dgem decide -u "https://<CLOUD_RUN_URL>/v1" --gcp-auth \
  -t templates/support_triage.json.tmpl -v 'ticket=Billing dispute'

# A gateway (routes to Vertex first, falls back to Cloud Run)
./bin/dgem decide -u "https://<your-dgem-gateway>/v1" --gcp-auth \
  -t templates/support_triage.json.tmpl -v 'ticket=General inquiry'
```

Persistent defaults go in environment variables or `.dgem.yaml` (project root or `~/.config/dgem/config.yaml`):

```bash
export DGEM_VERTEX_URL=<ENDPOINT_ID>   # or DGEM_URL=https://<CLOUD_RUN_URL>/v1
export DGEM_GCP_AUTH=true
```

```yaml
url: "https://<your-dgem-gateway>/v1"
gcp_auth: true
timeout: 120s
```

## Authentication

- `--gcp-auth` uses Application Default Credentials (`gcloud auth application-default login`) and mints the right
  token per target: an OAuth access token for Vertex AI, an ID token for Cloud Run and IAP-protected gateways.
- Static keys: `-k <token>` for self-hosted endpoints started with an API key.
- Debugging without tokens: `gcloud run services proxy <SERVICE> --region <REGION> --port 8080`, then
  `-u http://127.0.0.1:8080/v1`.
- Clients should retry `429`/`503` with backoff (`--http-retries N`): Vertex returns `429` while a replica is
  starting and the server returns `503` when its in-flight queue is full.

## Studio and MCP against a remote target

```bash
# Local Studio in front of your endpoints
./bin/dgem serve --port 8090 --gcp-auth --vertex-url <ENDPOINT_ID> -u https://<CLOUD_RUN_URL>/v1

# MCP (stdio) bridged to a hosted gateway, with automatic ADC auth
./bin/dgem mcp --remote https://<your-dgem-gateway>/mcp
```

More MCP and HTTP options: [Studio, MCP and HTTP API](../reference/studio-mcp-api.md). Routing between backends:
[Gateway and routing](gateway.md).

## Other hosts

Any host running the dgem serving container ([build it](public-images.md)) (a GCE VM, Kubernetes, a workstation) serves the same API on port 8080:
point `-u` at `http://<host>:8080/v1`. The GCE helper scripts (`make gce-deploy`, `make gce-teardown`) create and
delete a single-GPU VM; delete it as soon as you're done, since it bills while running.
