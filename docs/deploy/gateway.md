---
title: "Step 5: Gateway and Routing"
description: "Put one dgem gateway (dgem serve) in front of Vertex AI and Cloud Run: backend modes, automatic failover, per-request overrides on every surface, IAP access and deployment."
---

# Step 5: Gateway and Routing

The gateway (`dgem serve`) is what users and agents talk to. It serves Decision Studio, the HTTP API
(`/api/decide`, `/v1/systemone`; `/v1/chat/completions` is deprecated), the MCP server (`/mcp`) and traces, and routes each
request to a GPU backend. Locally it runs on `http://localhost:8090`; in the cloud it runs as the Cloud Run service
`dgemma-gateway` behind IAP (placeholder `https://<your-dgem-gateway>`).

## Backend modes

This is the reference table for backend selection; other pages link here.

| Mode | Sends requests to | When the primary is unavailable | Wake-up | Use it for |
| :--- | :--- | :--- | :--- | :--- |
| **`vertex_first`** (default) | Vertex AI dedicated endpoint | Fails over to Cloud Run (endpoint updating, scaled to 0, or erroring) | None while Vertex is warm | Studio, MCP agents, production APIs |
| `vertex` | Vertex AI only | Returns the error | None | Strict latency SLAs; testing the endpoint itself |
| `cloudrun` | Cloud Run GPU only | Holds and retries while the service wakes (`--wakeup-timeout`, default 10 min) | A few minutes after idle | Batch jobs, experiments, no-cost-when-idle use |
| `local` | Local `diffgemma` (`--local`) | Returns the error | None | Laptop development |

What each backend provides: [Production on Vertex AI](vertex.md#what-you-get) and
[Deploy on Cloud Run](cloud-run.md#what-you-get). `GET /api/backend-config` lists the backends the gateway is
configured for (`available_backends`); the Studio hides the backend switcher when only one is configured.

## Restrict backends and admin actions

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `--backends vertex_first,cloudrun,local` (`DGEM_BACKENDS`) | derived from the configured URLs | Explicit allow-list. Startup fails if a listed backend lacks its URL or `--default-backend` is not in the list. |
| `--enable-admin-api` (`DGEM_ADMIN_API=1`) | **off** | Allows `POST /api/backend-config` (gateway-wide default backend, Vertex URL, local URL) and `POST /api/vertex/deploy` / `teardown`. Off: these return `403`, and the Studio hides endpoint management. |
| `--allowed-vertex-endpoints <id,...>` (`DGEM_ALLOWED_VERTEX_ENDPOINTS`) | none | Extra Vertex endpoints a request may select with `vertex_url` / `X-DGem-Vertex-Url`. The configured endpoint is always allowed; any other returns `400`. Endpoints are matched by project, region and endpoint ID (and host for dedicated endpoints), so the same endpoint ID in another project doesn't match. |
| `--cors-origins <origin,...>` (`DGEM_CORS_ORIGINS`) | none | Browser origins (for example `https://studio.example.com`) that may call the gateway from another site. The embedded Studio is same-origin and needs none. Cross-origin `POST`s from any other site get `403`; `*` is refused. |
| `--allowed-hosts <host,...>` (`DGEM_ALLOWED_HOSTS`) | none | Extra `Host` header values accepted on loopback connections, for a reverse proxy on the same machine. Without it, a request reaching `127.0.0.1` with a non-loopback `Host` gets `403` (DNS rebinding protection, `/mcp` included). |

Requests naming an unknown backend, or one that is not enabled, get `400` with the list of available backends
(HTTP header, query, JSON body and MCP alike); a request without a backend uses the gateway default. The Studio's
backend menu is per browser and sends `X-DGem-Backend` with each request; only an admin-enabled gateway lets it
change the default for everyone.

Clients that are not browsers (curl, SDKs, MCP clients) send no `Origin` header and are unaffected by
`--cors-origins`. A web page on another site can neither read gateway responses nor send it `POST` requests unless
its origin is listed, so it can't use the gateway's cloud credentials through a user's browser.

Keep the admin API off on shared gateways: with it on, any user who passes IAP can change routing for all users
or deploy and tear down the Vertex endpoint with the gateway's service account. Manage endpoints with
[`deploy_vertex_endpoint.sh`](vertex.md) instead, or run a separate admin-only gateway.

## Choose a backend per request

| Surface | How |
| :--- | :--- |
| Decision Studio | Topbar **Backend Target** menu (also shows live Vertex replica state) |
| HTTP API | Header `X-DGem-Backend: vertex_first\|vertex\|cloudrun\|local`, query `?backend=...`, or JSON `"backend": "..."` |
| MCP | `"backend"` argument on `decide_policy`, `decide_custom_questions`, `locate_bounding_boxes` |
| CLI through a gateway | `-u https://<your-dgem-gateway>/v1 --gcp-auth` (gateway default applies) |
| Gateway default | `--default-backend` / `DGEM_DEFAULT_BACKEND` |

Every response reports where it ran: header `X-DGem-Backend-Used: vertex|cloudrun|local` (and `"backend_target"` in
JSON bodies, including MCP tool results).

```bash
curl -sS https://<your-dgem-gateway>/api/decide/support_triage \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" -H "X-DGem-Backend: vertex_first" \
  -d '{"variables": {"ticket": "Production database unreachable after certificate rotation."}}' -i \
  | grep -i x-dgem-backend-used
```

## Run a gateway locally

```bash
./bin/dgem serve --gcp-auth --vertex-url <ENDPOINT_ID> -u https://<CLOUD_RUN_URL>/v1   # both backends
./bin/dgem serve --local                                                               # laptop engine only
```

## Deploy the gateway on Cloud Run

```bash
GCP_PROJECT=<PROJECT> GCP_REGION=<REGION> GCP_PROJECT_NUMBER=<PROJECT_NUMBER> \
ALLOW_GROUP=<GROUP> \
DGEM_VERTEX_URL=<ENDPOINT_ID> DGEM_VERTEX_MODEL_ID=<MODEL_ID> \
DGEM_VERTEX_SA=dgemma-gpu-sa@<PROJECT>.iam.gserviceaccount.com \
DGEM_GATEWAY_HOSTS=<CUSTOM_DOMAIN> \
GATEWAY_TAG=candidate \
./scripts/deploy_cloudrun_gateway.sh
```

- Builds the gateway image (Go binary + Studio) with Cloud Build, tagged with the git commit, and deploys
  `dgemma-gateway` with `--min-instances=1 --no-cpu-throttling`, so its 4-minute Vertex health keepalive keeps
  running.
- `ALLOW_GROUP` (required) gets access through IAP / Cloud Run IAM; `DGEM_VERTEX_URL` is required unless
  `ALLOW_NO_VERTEX=1` (Cloud Run only).
- `DGEM_GATEWAY_HOSTS`: extra hostnames (for example a custom domain) the gateway treats as its own for IAM token
  handling. `*.run.app` and Vertex endpoints are recognised automatically.
- `DGEM_BACKENDS`, `DGEM_ALLOWED_VERTEX_ENDPOINTS` and `DGEM_ADMIN_API=1` are passed through when set
  ([restrictions](#restrict-backends-and-admin-actions)).
- When the Secret Manager secret `dgem-server-key` exists (`SERVER_KEY_SECRET` to name another), the script grants
  the gateway service account access and mounts it as `DGEM_SERVER_KEY`; the gateway then sends it as `X-DGem-Key`
  to the serving image ([service key](vertex.md#1-prerequisites)). Callers' own `X-DGem-Key` headers are never
  forwarded.
- `GATEWAY_TAG`: deploy as a tagged revision with **no traffic**, verify it at `https://<tag>---<gateway-host>`,
  then `gcloud run services update-traffic dgemma-gateway --to-tags=<tag>=100`. Roll back by moving traffic to
  the previous revision.

Verify a new gateway revision before moving traffic:

1. `GET /health` and `GET /api/backend-config` (expected `available_backends`, Vertex `state: deployed`).
2. One `/api/decide` call per backend, checking `X-DGem-Backend-Used`.
3. MCP `tools/list` and the `get_health_and_gpu_status` tool.
4. After moving traffic, repeat through the custom domain (without a token IAP should redirect to sign-in).

## Access for people and agents

- **Browser:** sign in through IAP as a member of `<GROUP>`.
- **CLI / scripts:** `--gcp-auth` sends a Google ID token from Application Default Credentials.
- **MCP clients:** `dgem mcp --remote https://<your-dgem-gateway>/mcp` bridges stdio to the gateway with automatic
  ADC auth. Other options: [Studio, MCP and HTTP API](../reference/studio-mcp-api.md).
- Observability: every gateway decision is traced (`dgem.gateway.decide` and child spans); see
  [Observability](../operate/observability.md).

## Next

[Latency and capacity](../operate/latency-capacity.md) and the [operations runbook](../operate/runbook.md).
