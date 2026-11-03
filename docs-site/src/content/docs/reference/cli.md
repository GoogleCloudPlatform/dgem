---
title: "CLI, HTTP Gateway & MCP Reference"
description: "Complete command-line flag reference for dgem (--vertex-url, dgem serve --default-backend vertex_first --cascade-model gemini-3.8-flash, dgem systemone serve), HTTP Gateway proxy routes (/api/decide, /v1/systemone), and Model Context Protocol (MCP) tool parameters."
---

This reference documents the global `dgem` CLI flags, the `dgem serve` Decision Studio & Gateway server options, the `dgem systemone serve` tournament adapter, the `/v1/systemone` and `/api/decide` HTTP routes, and the Model Context Protocol (`MCP`) tool schemas.

---

## 1. Global CLI Flags (`dgem`)

All `dgem` subcommands (`decide`, `ask`, `locate`, `template`, `serve`, `systemone`, `mcp` and the `bench*` harnesses) share the following global flags and `DGEM_*` environment variables:

| Flag | Env Var | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--vertex-url`** | `DGEM_VERTEX_URL` | `""` | Vertex AI dedicated endpoint ID or full invoke URL. A bare ID expands to `https://<endpoint-id>.<region>-<project-number>.prediction.vertexai.goog/v1/projects/<project-number>/locations/<region>/endpoints/<endpoint-id>/invoke/v1` and needs `GCP_PROJECT_NUMBER` (region from `GCP_REGION`, default `us-central1`). When specified on CLI commands (`dgem decide --vertex-url <endpoint-id> --gcp-auth`), `dgem` mints an OAuth2 `cloud-platform` access token (`gcloud auth print-access-token`) and routes directly to `/invoke/v1/*`. |
| **`-u, --url`** | `DGEM_URL` | `http://127.0.0.1:8080/v1` | Base URL of the upstream server (Local Apple Silicon `diffgemma`, Serverless Cloud Run `dgemma`, or `dgemma-gateway`). |
| **`-m, --model`** | `DGEM_MODEL` | `diffgemma-26b-a4b-it-q4` | Model identifier (`/model` or `/mnt/gcs/dgemma` on Vertex AI / Cloud Run). |
| **`--gcp-auth`** | `DGEM_GCP_AUTH` | `false` | Automatically mint Google Cloud authentication tokens (`gcloud auth print-access-token` for Vertex AI `/invoke/*` and Stage 2 Gemini `generateContent`; `gcloud auth print-identity-token` for Cloud Run). |
| **`-k, --token`** | `DGEM_TOKEN` | `""` | Explicit Bearer token override for secured endpoints. |
| **`-s, --stats`** | `DGEM_STATS` | `false` | Print comprehensive GPU timing, KV cache hit rate, restricted-softmax probabilities, and Shannon entropy ($H$ & $\tilde{H}$). |
| **`--timeout`** | `DGEM_TIMEOUT` | `2m` | HTTP client timeout duration. |
| **`--http-retries`** | — | `0` (`serve`, `mcp`: 3) | Retry HTTP 429/503 with backoff. |

### CLI Examples Against Vertex AI Dedicated Endpoint (`--vertex-url`)

```bash
# 1. One decision on a Vertex AI dedicated endpoint (bare IDs need GCP_PROJECT_NUMBER):
./bin/dgem decide --vertex-url <endpoint-id> --gcp-auth \
  -t templates/support_triage.json.tmpl \
  -v 'ticket=I was billed twice for my annual renewal this morning!' \
  --stats

# 2. The 30-case multi-domain decision suite on a Vertex AI Dedicated Endpoint:
./bin/dgem bench --vertex-url <endpoint-id> --gcp-auth \
  -d benchmarks/eval_dataset.jsonl \
  -o bench_vertex.json
```

---

## 2. `dgem serve` Flags (Decision Studio, HTTP Gateway & `/mcp`)

`dgem serve` starts the unified Web Studio UI, HTTP Gateway REST API, `/v1/systemone` proxy, and Streamable HTTP MCP server:

```bash
./bin/dgem serve \
  --default-backend vertex_first \
  --vertex-url <endpoint-id> \
  --cascade-model gemini-3.8-flash \
  -u "https://<CLOUD_RUN_URL>/v1" \
  --gcp-auth
```

| Flag | Env Var | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--default-backend`** | `DGEM_DEFAULT_BACKEND` | **`vertex_first`** | Default upstream GPU routing policy when a request does not specify `X-DGem-Backend` or `backend`:<br/>• **`vertex_first`** *(Recommended)*: Routes to the warm Vertex AI Dedicated Endpoint (`--vertex-url`) and automatically falls back to Serverless Cloud Run GPU (`-u`) if Vertex is updating or scaled to zero.<br/>• **`vertex`**: Strictly pins requests to the Vertex AI Dedicated Endpoint (`/invoke/*`).<br/>• **`cloudrun`**: Strictly pins requests to Serverless Cloud Run GPU (`dgemma`). |
| **`--vertex-url`** | `DGEM_VERTEX_URL` | `""` | Target Vertex AI Dedicated Endpoint ID or `/invoke/v1` URL used by `vertex_first` and `vertex` routing modes. |
| **`--cascade-model`** | `DGEM_CASCADE_MODEL` | **`gemini-3.8-flash`** | Default Vertex AI Gemini model for Stage 2 Escalation Cascades (`gemini-3.8-flash` or `gemini-3.5-flash-lite`). |
| **`--cascade-models`** | `DGEM_CASCADE_MODELS` | **`gemini-3.8-flash,gemini-3.5-flash-lite`** | Comma-separated list of selectable Stage 2 Vertex AI Gemini 3.x models exposed in `GET /api/backend-config` and the Web Studio. |
| **`--systemone-mode`** | `DGEM_SYSTEMONE_MODE` | `adapter` | Mode for the `/v1/systemone` route: `adapter` (bracket tournaments and multi-slot batching, default) or `passthrough` (direct proxy). |
| **`--systemone-temperature`** | `DGEM_SYSTEMONE_TEMP` | `1.0` | Post-hoc slot logit temperature scaling $T^*$ for `/v1/systemone` decisions. |
| **`-u, --url`** | `DGEM_URL` | `http://127.0.0.1:8080/v1` | Upstream Serverless Cloud Run GPU `/v1` URL used by `cloudrun` routing and `vertex_first` failover. |
| **`-p, --port`** | `PORT` | `8090` | HTTP listener port (the flag overrides `PORT`). |
| **`--host`** | — | `127.0.0.1` (`0.0.0.0` on Cloud Run/Vertex) | Interface to bind. The default only accepts connections from this machine (Studio at `http://localhost:8090`); pass `--host 0.0.0.0` to accept other machines, and put the gateway behind IAP/IAM. Cloud Run (`K_SERVICE`) and Vertex (`AIP_HTTP_PORT`) default to `0.0.0.0`; the gateway image passes it explicitly. |
| **`--local`, `--local-url`** | `DGEM_SERVE_LOCAL`, `DGEM_LOCAL_URL` | off | Serve against a local `diffgemma` engine only (backend `local`, no cloud failover). |
| **`--backends`** | `DGEM_BACKENDS` | derived | Backend allow-list (`vertex_first`, `vertex`, `cloudrun`, `local`); validated at startup. |
| **`--enable-admin-api`** | `DGEM_ADMIN_API` | off | Allow `POST /api/backend-config` and `/api/vertex/deploy`, `/api/vertex/teardown`. |
| **`--allowed-vertex-endpoints`** | `DGEM_ALLOWED_VERTEX_ENDPOINTS` | none | Extra Vertex endpoints selectable per request; others are rejected. |
| **`--cors-origins`** | `DGEM_CORS_ORIGINS` | none | Browser origins allowed to call the gateway cross-origin. Others get no CORS headers, and their `POST`s get `403`. |
| **`--iap-audience`** | `DGEM_IAP_AUDIENCE` | none | Verify the IAP JWT (`X-Goog-IAP-JWT-Assertion`: signature, audience, expiry) on every request except `GET /health`; the caller's identity comes from the verified token. Cloud Run IAP: `/projects/<PROJECT_NUMBER>/locations/<REGION>/services/<SERVICE>`. Missing or invalid token: `401`. |
| **`--api-keys`** | `DGEM_GATEWAY_KEYS` | none | Comma-separated API keys (rotate with `new,old`) accepted in `X-DGem-Api-Key`, for self-hosted gateways without IAP. Callers send `DGEM_GATEWAY_KEY`: the `dgem` CLI only to the gateway in `DGEM_GATEWAY_URL`, `dgem mcp --remote` to its remote. With `--iap-audience` too, either is enough. Prefer the env var over the flag. |
| **`--allowed-hosts`** | `DGEM_ALLOWED_HOSTS` | none | Extra `Host` values accepted on loopback connections (reverse proxy on the same machine). Others get `403`. |
| **`--wakeup-timeout`** | — | `10m` | How long `cloudrun` requests are held and retried while the service wakes from zero. |

---

## 3. `dgem systemone serve` (Decision Index Adapter)

`dgem systemone serve` runs a standalone HTTP adapter that implements the `/v1/systemone` decision protocol (`POST /v1/systemone`, also used by the Decision Index benchmark kit). It splits choices with more than 26 options into 2-stage bracket tournaments and batches more than 8 questions across forward passes, sending sub-requests to an upstream [DiffusionGemma](https://gemma.ai/models/diffusiongemma/) server over `/v1/chat/completions`.

```bash
# In front of a local structured server / vLLM:
dgem systemone serve --port 8080 --upstream http://127.0.0.1:8081/v1

# In front of a Vertex AI Dedicated Endpoint (a GCP access token is added automatically):
dgem systemone serve --port 8095 \
  --upstream "https://<endpoint-id>.<region>-<project-number>.prediction.vertexai.goog/v1/projects/<project-id>/locations/<region>/endpoints/<endpoint-id>/invoke/v1"

# Require a Bearer key on incoming requests:
dgem systemone serve --port 8080 --api-key "my-secret-key"
```

The serving container can run the adapter itself: set `ROLE=decision-index` (optionally `TEMPERATURE`, `CANVAS`)
and the container serves the adapter on its port, in front of its own structured server.

| Flag | Env Var | Default | Description |
| :--- | :--- | :--- | :--- |
| **`-p, --port`** | `PORT`, then `SYSTEMONE_PORT` | `8080` | Port to listen on. |
| **`--host`** | — | `127.0.0.1` (`0.0.0.0` on Cloud Run/Vertex) | Interface to bind. The default only accepts connections from this machine; pass `--host 0.0.0.0` to accept others (set `--api-key` then). Cloud Run (`K_SERVICE`) and Vertex (`AIP_HTTP_PORT`) default to `0.0.0.0`, and the serving image's entrypoint passes it explicitly. |
| **`--upstream`** | `SYSTEMONE_UPSTREAM_URL`, then `UPSTREAM_DGEMMA_URL` | `-u` / `--url` if set, else `http://127.0.0.1:8081/v1` | Upstream `/v1` base URL (local server, Cloud Run, or a Vertex `.../invoke/v1` URL). |
| **`--temperature`** | — | `1.0` | Post-hoc slot temperature scaling $T^*$ (`1.0` = unscaled). |
| **`--max-slots`** | — | `8` | Maximum questions per forward pass before batching. |
| **`--bracket-size`** | — | `20` | Maximum options per round-1 tournament bracket. |
| **`--api-key`** | `SYSTEMONE_API_KEY`, then `API_KEY` | `""` (open) | If set, requests must send `Authorization: Bearer <key>`. |
| **`--null-prior-debias`** | — | `false` | Research only: divide out the positional option-`A` prior. Not recommended in production; see [hesitation-gating](/dgem/confidence/overview/). |
| **`--prior-alpha`** | — | `0.50` | Research only: exponent for null-prior de-biasing. |
| **`--dual-mirror`** | — | `false` | Research only: also read a reversed option ordering. Not recommended in production. |
| **`--naive-limits`** | — | `false` | Reproduce the naive 26-option / 10-question rejections (benchmark ablation only). |
| **`--prompt-layout`** | — | `document_first` | How the upstream prompt is laid out: `document_first` (the state, then the questions, both in the user turn) or `schema_first` (the questions as the system prompt, the layout before v0.2.0). Serving images from v0.2.0 default to `document_first` themselves (`DEFAULT_LAYOUT`); older ones ignore the field. In our tests `document_first` improved accuracy on catch-all intent and hallucination-detection questions, at some cost in out-of-scope recall ([Prompt layout](/dgem/policies/prompt-layout/)). |
| **`--catch-all`** | — | `off` | Wide-option catch-all handling (`off`, `final`, `both`, `verify`). `final` keeps "none of the listed"-style options out of round-1 brackets and adds them to the final. Opt-in: it trades out-of-scope recall for in-scope accuracy. |
| **`--option-order`** | — | `alpha` | How each question's options are labelled A, B, C…: `alpha` sorts the option keys (the behaviour so far), `source` keeps the order the request lists them in, through every bracket round. `source` is for measuring first-shown bias; keep `alpha` unless you have measured otherwise. |
| **`--labels`** | — | `az` | Server label set for choice options. `az` (A–Z) brackets above 26 options. `az_aa` / `az_lower` (52 labels) read up to 52 options in one pass and bracket in 52-wide groups above that; they need a serving image with label sets. Experimental. |
| **`--noul-mode`** | — | `noul` | How yes/no questions are read: `noul`, or `choice` (a 2-option yes/no choice with the true/false criteria as descriptions). Opt-in: it helps hallucination-style questions and costs a little on other yes/no families. |

---

## 4. HTTP Gateway Endpoints (`/api/decide`, `/v1/systemone`)

`/api/decide` also accepts a prompt layout as JSON `"layout": "document_first" | "schema_first"`, header `X-DGem-Layout` or `?layout=`; the server reports the layout it used in `diagnostics.layout` ([Prompt layout](/dgem/policies/prompt-layout/)). `/v1/systemone` and the deprecated `/v1/chat/completions` pass a `layout` key in the body or schema through unchanged.

Every inference endpoint on `dgem serve` (`https://<your-dgem-gateway>`) accepts backend selection via HTTP header `X-DGem-Backend: vertex_first | vertex | cloudrun | local`, query parameter `?backend=vertex_first`, or JSON body field `"backend": "vertex_first"`, and returns the **`X-DGem-Backend-Used: vertex | cloudrun | local`** response header (`"backend_target"` in JSON bodies).

| Route | Method | Description |
| :--- | :---: | :--- |
| **`/api/decide` & `/api/decide/{template}`** | `POST` | Renders a named or inline (`custom_template`) `.json.tmpl` policy with `variables`, executes Stage 1 [`DiffusionGemma`](https://gemma.ai/models/diffusiongemma/) readout on `vertex_first` / `vertex` / `cloudrun`, and optionally runs the **Stage 2 Gemini Cascade** (`cascade_mode`: `"off" \| "entropy" \| "on_miss"`, `cascade_threshold`: `0.35`, `cascade_threshold_mode`: `"nats" \| "normalized"`, `cascade_model`: `"gemini-3.8-flash"`, `stage2_prior`: `"soft" \| "full" \| "none"`, `temperature`: a number or `{"noul": T, "choice": T, "score": T}`). |
| **`/api/locate`** | `POST` | Guided locate: `{"image": "<data-uri-or-public-url>", "target": "the checkout button", "mask": false, "gemini_model": "gemini-3.8-flash", "thinking": "low", "skip_h": 0, "hint": false, "backend": "vertex_first"}`. dgem answers presence and 3×3 cell, then Gemini 3.x (LOW thinking) returns `box_1000` / `box_pct` (`path: gemini` or `gemini_absent`). Opt-ins: `skip_h` > 0 (0.16 recommended on photos and mobile UI, not on defect images) returns `path: skipped_absent` without Gemini when dgem is confidently absent; `hint: true` passes dgem's cell to Gemini (neutral on average in our tests). With `mask: true` and `DGEM_SAM_URL` set, a SAM service adds `mask`; otherwise `mask_note` explains why not. Returns per-stage latency (`dgem.ms`, `gemini.ms`, `sam_ms`, `total_ms`). Images must be data: URIs or public http(s) URLs. Spans: `dgem.gateway.locate` → `dgem.locate` → `dgem.locate.dgem` / `.gemini` / `.sam`. |
| **`/v1/systemone`** | `POST` | `SystemOne` / `JevBench` decisions (`{"state": ..., "questions": {...}, "images": [...]}`). JSON bodies go through the adapter (`--systemone-mode adapter`, the default: brackets for choices with more than 26 options, batching past 8 questions), which sends `images` (data: URIs or public http(s) URLs; local paths are rejected with `400`) with every sub-request. `multipart/form-data` bodies (a `request` JSON part plus image file parts) and `?raw=1` pass through unchanged to `/invoke/v1/systemone` on Vertex AI or `/v1/systemone` on Cloud Run GPU. |
| **`/v1/chat/completions`** | `POST` | **Deprecated** (responses carry `Deprecation` and `Link: </v1/systemone>; rel="successor-version"`). Decision-only proxy: the system message is the question schema and the user message the JSON state; anything else is rejected upstream with `400`. Kept for `dgem decide -u https://<your-dgem-gateway>/v1`; new callers use `/api/decide` or `/v1/systemone`. |
| **`/v1/raw/chat/completions`** | `POST` | **Removed**: returns `410 Gone`. dgem serves decisions, not free-form generation. |
| **`/api/backend-config`** (live `vertex_status`, `available_backends`, `admin_api`), **`/api/vertex/deploy`**, **`/api/vertex/teardown`** | `GET` / `POST` | Live Vertex endpoint status (`GET`, always allowed). `POST` changes gateway-wide settings or deploys/tears down the endpoint and requires `--enable-admin-api` (otherwise `403`). |

---

## 5. Model Context Protocol (`MCP`) Tool Parameters (`POST /mcp` & `dgem mcp`)

The MCP inference tools (`decide_policy`, `decide_custom_questions`, `locate_object` and `locate_bounding_boxes`) accept the following backend routing parameters. The two decide tools also accept the Stage 2 Gemini Cascade and taxonomy expansion parameters:

| MCP Argument | Type | Allowed Values / Default | Description |
| :--- | :--- | :--- | :--- |
| **`backend`** | `string` | `"vertex_first"` *(default)* \| `"vertex"` \| `"cloudrun"` \| `"local"` | Selects the execution target (`vertex_first` routes to warm Vertex AI Dedicated Endpoint with automatic Cloud Run failover; `local` is a `diffgemma` engine on your machine and is the default under `dgem mcp --local`). Only configured backends are accepted; any other value returns `backend "…" is not enabled` with the available list. The tool schema lists exactly the backends this server has (`enum`) and names its default, so agents see the right choices. Omit it to use the default. `locate_bounding_boxes` does not support `local`. |
| **`vertex_url`** | `string` | `""` *(optional)* | Vertex AI Dedicated Endpoint ID or `/invoke/v1` URL override. Must be the configured endpoint or one listed in `--allowed-vertex-endpoints` on `dgem serve`; any other returns `vertex_url … is not allowed`. |
| **`layout`** | `string` | `""` *(template or server default)* \| `"document_first"` \| `"schema_first"` | Decide tools only. Prompt layout; see [Prompt layout](/dgem/policies/prompt-layout/). |
| **`cascade_mode`** | `string` | `"off"` *(default)* \| `"entropy"` \| `"on_miss"` | Stage 2 Gemini Cascade trigger policy (`"entropy"` escalates when Stage 1 Shannon entropy $H \ge$ `cascade_threshold`; `"on_miss"` escalates slots that disagree with `expected_answers`). |
| **`cascade_threshold`** | `number` | `0.35` *(default, in nats)* | Shannon entropy threshold $\tau$ in nats for `"entropy"` escalation. |
| **`cascade_model`** | `string` | `"gemini-3.8-flash"` *(default)* | Stage 2 Vertex AI Gemini model (`"gemini-3.8-flash"` or `"gemini-3.5-flash-lite"`). |
| **`stage2_prior`** | `string` | `"soft"` *(default)* \| `"full"` \| `"none"` | Decide tools only. How much of Stage 1's answer the Stage-2 Gemini cascade sees (see the [authoring guide](/dgem/policies/authoring/)). |
| **`temperature`**, **`temperature_by_type`** | `number`, `object` | none | Decide tools only. Post-hoc temperature for every answer, or per question type (`{"noul": 3, "choice": 1.3}`); overrides the template's `"temperature"`. |
| **`expected_answers`** | `object` | `{}` *(optional)* | Map of slot id to expected value, used by `cascade_mode: "on_miss"`. |
| **`suggest_expansions`** | `boolean` | `false` | Adds an `other_unclassified` option to `choice` slots and proposes new options when a slot is unclassified or hesitant ([taxonomy discovery](/dgem/policies/taxonomy-discovery/)). |
| **`expansion_entropy`** | `number` | `0.35` *(nats)* | Entropy threshold on `choice` slots that triggers an expansion proposal. |

The response fields of the decide tools are described in [Studio, MCP and HTTP API](/dgem/reference/studio-mcp-api/#33-reading-decide-tool-results).

The backend allow-list and endpoint restrictions (`--backends` / `DGEM_BACKENDS`, `--allowed-vertex-endpoints` /
`DGEM_ALLOWED_VERTEX_ENDPOINTS`, `--enable-admin-api`) are settings of `dgem serve`, so they apply to a gateway's
`POST /mcp` and to `dgem mcp --remote`. Stdio `dgem mcp` runs with your own credentials and offers every backend it
has an endpoint for: Vertex from `--vertex-url` or `DGEM_VERTEX_URL` (a bare endpoint ID also needs
`GCP_PROJECT_NUMBER`; a full `/invoke` URL doesn't), Cloud Run from `-u` or `DGEM_REMOTE_URL`, and `local` with
`--local`.

---

## 6. Environment Variables Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| **`DGEM_VERTEX_URL`** | `""` | Target Vertex AI Dedicated Endpoint ID or `/invoke/*` URL. |
| **`DGEM_VERTEX_ENDPOINT_ID`** | `""` | Dedicated Endpoint ID override. |
| **`DGEM_VERTEX_MODEL_ID`** | `""` | Target Vertex AI Model ID deployed by `/api/vertex/deploy`. |
| **`DGEM_VERTEX_SA`** | `""` | Dedicated Service Account for Vertex AI model deployment. |
| **`DGEM_DEFAULT_BACKEND`** | `vertex_first` | Default backend selector (`vertex_first`, `vertex`, `cloudrun`, `local`). |
| **`DGEM_CASCADE_MODEL`** | `gemini-3.8-flash` | Default Gemini model for Stage 2 escalation. |
| **`DGEM_CASCADE_MODELS`** | `gemini-3.8-flash,gemini-3.5-flash-lite` | List of enabled Gemini cascade models. |
| **`DGEM_SAM_URL`**, **`DGEM_SAM_TOKEN`** | `""` | Optional SAM service for `locate` masks: `POST {"image", "box_pct"}` → `{"png_base64", "polygon_pct", "score"}`; the token is sent as a Bearer header. Unset: masks are unavailable. |
| **`DGEM_GATEWAY_HOSTS`** | `""` | Comma-separated extra hostnames recognized as remote GCP endpoints for token injection. |
| **`DGEM_GATEWAY_KEY`** | `""` | API key for a gateway started with `--api-keys`, sent as `X-DGem-Api-Key` (the first entry of a comma-separated list). The CLI sends it only to `DGEM_GATEWAY_URL`; `dgem mcp --remote` sends it to its remote. |
| **`DGEM_GATEWAY_URL`** | `""` | The gateway `DGEM_GATEWAY_KEY` belongs to. The CLI sends the key only to URLs with the same scheme, host and port, and sends none while this is unset. If it names another gateway than `--remote`, `dgem mcp --remote` sends no key. |
| **`DGEM_REMOTE_URL`** | `""` | Upstream `/v1` URL used by plain `dgem mcp` (no `--local` / `--remote`); overrides `-u`. |
| **`DGEM_MCP_LOCAL`** | `""` | Set to `1` or `true` to behave like `dgem mcp --local`. |
| **`DGEM_BACKENDS`**, **`DGEM_ALLOWED_VERTEX_ENDPOINTS`**, **`DGEM_ADMIN_API`** | `""` | `dgem serve` only: backend allow-list, extra selectable Vertex endpoints, and admin API ([Gateway and routing](/dgem/deploy/gateway/#restrict-backends-and-admin-actions)). Not read by stdio `dgem mcp`. |
| **`GCP_PROJECT`** | `""` | Google Cloud project ID (falls back to GCP metadata server). |
| **`GCP_PROJECT_NUMBER`** | `""` | Google Cloud numeric project ID (detected automatically if on GCP). |
| **`GCP_REGION`** | `us-central1` | Default Google Cloud region for services and endpoints. |

---

## 7. Commands

### `dgem decide`
Executes single-pass discrete diffusion slot readout (or multi-stage conditional DAG execution when a template declares `depends_on` and `ask_if`).

```bash
dgem decide [flags]
```

#### Flags
* `-t`, `--template string`: Path to a Go policy template file (`.json.tmpl`).
* `-v`, `--var stringArray`: Template variables in `key=value` format (can be specified multiple times).
* `-d`, `--data string`: Path to a JSON file containing variables.
* `-f`, `--format string`: Output format: `table` (default) or `json`.
* `-I`, `--image stringArray`: Attach local image file path or remote image URL (can be specified multiple times for video frame sequences). PNG, JPEG, WebP or GIF; SVG is refused with a clear error because the model server can't decode it (convert it first, e.g. `rsvg-convert -o image.png image.svg`).
* `--temperature T`: Post-hoc temperature for every answer; 0 uses the template's `"temperature"`. See [calibrate your policy](/dgem/confidence/calibrate-your-policy/#5-temperature-scaling-optional).
* `--layout document_first|schema_first`: Prompt layout for this decision. Empty uses the template's `layout`, or else the server default (`document_first` from serving v0.2.0). See [Prompt layout](/dgem/policies/prompt-layout/).
* `--suggest-expansions`: Dynamically inject an `other_unclassified` catch-all option into `choice` slots (if absent) and propose new `{"name", "description"}` options when an item resolves to `other*` or exceeds `--expansion-entropy` (see [Unclassified Grouping & Taxonomy Discovery](/dgem/policies/taxonomy-discovery/)).
* `--expansion-entropy float`: Shannon entropy threshold in nats on `choice` slots to trigger taxonomy expansion proposals (default: `0.35`).
* `--dual-mirror` (research only): read a reversed copy of each choice question in the same pass. In our tests it lowered accuracy with lettered options and added little over hesitation; not recommended in production.
* `--null-prior-debias` (research only): divide out the content-free option-`A` habit (`--prior-alpha` sets the strength). In our tests it helped one suite and hurt another; not recommended in production.
* `--schema string`: Path to a raw JSON schema file (skips template engine).
* `--state string`: Raw JSON state string or file path.

#### Example 1: Multi-question triage against a Cloud Run service
```bash
./bin/dgem decide \
  -u "https://<CLOUD_RUN_URL>/v1" \
  --gcp-auth \
  -t templates/support_triage.json.tmpl \
  -v 'ticket=Emergency: production database cluster down!' \
  --stats
```

```text
QUESTION         | TYPE       | VALUE / CHOICE       | CONFIDENCE | STDERR     | AGREEMENT 
----------------------------------------------------------------------------------------
sentiment        | score      | furious              | 99.8%      | ±0.0000    | 1.00      
team             | choice     | engineering          | 99.9%      | ±0.0000    | 1.00      
urgent           | boolean    | yes                  | 100.0%     | ±0.0000    | 1.00      
```

#### Example 2: Conditional questions (`depends_on` / `ask_if`)
Questions with dependencies run in stages; a question whose `ask_if` condition is not met is skipped.

```bash
./bin/dgem decide -t templates/secops_conditional_dag.json.tmpl \
  -v 'alert_payload=Unusual IAM key creation followed by snapshot sharing to an external account.'
```

#### Example 3: Multimodal Visual Inspection (`--image` / `-I`)
```bash
./bin/dgem decide -t templates/multimodal/ui_design_review.json.tmpl \
  -I fixtures/ui_component.png \
  -v 'component=CheckoutCard' \
  --stats
```


### `dgem bench-calibration`
Runs the **50-case public-dataset calibration, guardrail and entropy-gated cascade suite** across 11 public datasets (`ChaosNLI`, `ANLI-R3`, `AgentDrift`, `deepset/prompt-injections`, `LLM-AggreFact`, `MS MARCO`, `Banking77`, `CLINC150`, `GoEmotions`, `CivilComments`, `Financial-PhraseBank`).

```bash
dgem bench-calibration [flags]
```

#### Key Flags
* `-d`, `--dataset string`: Path to calibration JSONL dataset (default: `benchmarks/calibration_suite.jsonl`).
* `-c`, `--category string` / `-T`, `--tier string`: Filter cases by category or tier.
* `-w`, `--workers int`: Concurrent request workers (default: `4`).
* `-o`, `--output string`: Path to write detailed JSON telemetry receipt (including `entropy`, `vocab_cardinality`, `normalized_entropy`, and `top_probabilities`).
* `--normalize-entropy`: Gate escalation using **Cardinality-Normalized Epistemic Entropy** ($\tilde{H}_m = H_m / \ln|\mathcal{V}_m| \in [0, 1]$) instead of raw Shannon entropy $H_m$ in nats.
* `--cascade-from string`: Hydrate Stage-1 results offline from an existing JSON receipt written with `-o` (or `live` to run Stage 1 now) and execute Stage-2 escalation only on items at or above the threshold.
* `--cascade-threshold float`: Entropy threshold for escalation (default: `0.35` nats raw, or `0.16` when paired with `--normalize-entropy`).
* `--vertex-model string`, `--vertex-project string`: Vertex AI Gemini model for Stage-2 escalation (for example `gemini-3.8-flash`) and its project (defaults to `$GCP_PROJECT` or the gcloud config). Without `--cascade-from`, `--vertex-model` evaluates the whole suite on Gemini.
* `--cascade-self-think int`: **Intra-Model Self-Cascade**: instead of calling an external model, re-invoke the **same [`DiffusionGemma`](https://gemma.ai/models/diffusiongemma/) endpoint** with `"think": <tokens>` (e.g., `256`) and the Pass-1 prior distribution block when $\tilde{H}_m \ge \tau$.

#### Examples
```bash
# 1. Run the 50-case calibration suite
./bin/dgem bench-calibration \
  -u "<CLOUD_RUN_URL>/v1" --gcp-auth -w 4 \
  -o calibration.json

# 2. Escalate hesitant items from that receipt to Gemini
./bin/dgem bench-calibration \
  --cascade-from calibration.json \
  --normalize-entropy \
  --cascade-threshold 0.16 \
  --vertex-model gemini-3.8-flash --vertex-project <PROJECT> \
  -o calibration_cascade.json

# 3. Self-cascade: re-ask hesitant items on the same model with a reasoning trace
./bin/dgem bench-calibration \
  -u "<CLOUD_RUN_URL>/v1" --gcp-auth \
  --normalize-entropy \
  --cascade-threshold 0.16 \
  --cascade-self-think 256 \
  -o calibration_self_cascade.json
```


### `dgem bench`
Runs the **30-case multi-domain decision suite** evaluating joint 3-slot decisions across `support_triage` (10 cases), `code_review` (10 cases), and `security_incident` (10 cases) (`benchmarks/eval_dataset.jsonl`).

```bash
dgem bench [flags]
```

#### Key Flags
* `-d`, `--dataset string`: Path to evaluation dataset (default: `benchmarks/eval_dataset.jsonl`).
* `-w`, `--workers int`: Number of parallel evaluation workers (default: `1`, use `4` on Cloud Run GPU).
* `-M, --mode both`: also run full autoregressive text generation for comparison (much slower).
* `-o`, `--output string`: Save a structured JSON receipt.

```bash
./bin/dgem bench -u "<CLOUD_RUN_URL>/v1" --gcp-auth -w 4 -o bench.json
```


### `dgem bench-intents`
Runs high-cardinality intent classification and out-of-scope (`oos`) detection on **`PolyAI/banking77`** (3,080 test items) and **`DeepPavlov/clinc150`** (5,500 test items).

```bash
dgem bench-intents [flags]
```

#### Key Flags
* `-d`, `--dataset string`: Target benchmark (`banking77` or `clinc150`).
* `-F`, `--full`: Download and evaluate the complete upstream Hugging Face test splits (`3,080` / `5,500` items) instead of the 30-item curated smoke subsets.
* `-w`, `--workers int`: Concurrent worker pool size (default: `4`, recommended: `16` on Cloud GPU).
* `-o`, `--output string`: Output JSON receipt path.

```bash
./bin/dgem bench-intents --dataset banking77 --full --workers 16 \
  -u "<CLOUD_RUN_URL>/v1" --gcp-auth \
  -o intents_banking77_full.json
```


### `dgem locate`
Locates an object or UI element in an image with a guided pipeline: one dgem pass (presence + 3×3 cell), then a Gemini 3.x box at LOW thinking. `--skip-h 0.16` skips Gemini when dgem is confidently absent (recommended on photos and mobile UI, not on defect images); `--hint` passes dgem's cell to Gemini. `--mask` adds a SAM mask when `DGEM_SAM_URL` is set. See [what dgem can do with images](/dgem/policies/images/).

```bash
./bin/dgem locate --vertex-url <ENDPOINT_ID> -I screenshot.png --target "the checkout button"
./bin/dgem locate --vertex-url <ENDPOINT_ID> -I photo.jpg --target "the dog on the left" --gemini-model gemini-3.5-flash-lite -f json
```

Flags: `-I/--image`, `--target`, `--mask`, `--gemini-model`, `--thinking low|medium|high|default`, `--skip-h` (0 = never skip, the default), `--hint`, `-f table|json`. Gemini needs Application Default Credentials and `GOOGLE_CLOUD_PROJECT`.

### `dgem bench-bbox`
Runs the **single-pass bounding-box suite** (`benchmarks/bbox_suite.jsonl`, `fixtures/bbox/`) or any custom image directory (`--dir`). For each image it reads `[ymin, xmin, ymax, xmax]` in one pass and reports:
- argmax and softmax-expectation boxes, with mIoU and Acc@0.5/0.75 and case-level bootstrap 95% intervals;
- image-free baselines (a fixed centre box, the leave-one-out mean box and, with `--variants blank`, the model's own box on a blank image) and the model's paired lift over each;
- the edge error split into rounding (quantization) and localization, plus signed bias per edge;
- whether per-edge entropy predicts wrong edges (AUROC);
- paired occlusion deltas and run-to-run stability.

A slot the model does not return counts as a failure. See [what dgem can do with images](/dgem/policies/images/).

```bash
dgem bench-bbox [flags]
```

#### Key Flags
* `-d`, `--dataset string`: Path to the bounding-box JSONL suite (default: `benchmarks/bbox_suite.jsonl`).
* `--dir string`: Custom directory containing `.png`/`.jpg`/`.webp` images (and optional `manifest.jsonl` or `index.txt`) for ad-hoc runs.
* `--target string`: Default target object description when running `--dir` without a manifest.
* `--repeat int`: Run every case N times to measure run-to-run stability (default 1).
* `--variants string`: Probe variants, comma-separated or `all`:
  * `reversed`: the coordinate options in reverse order.
  * `digits9`, `digits9_reversed`: 9-level digit labels, forward or descending.
  * `hflip`, `vflip`: the image mirrored.
  * `pad_right`, `pad_left`, `pad_bottom`: the canvas extended by 50%.
  * `blank`: a blank image of the same size.

  Image transforms report consistency with the original box mapped through the same transform.
* `-w`, `--workers int`: Concurrent requests (default 4). `--samples int`: noise draws per request (default 1).
* `--from-receipt string`: Re-analyze an existing receipt (adds baselines, intervals and the error split) without calling a server.
* `--engine dgem|gemini|predictions`: who produces the boxes. `gemini` asks a Gemini 3.x model (`--gemini-model`, default `gemini-3.8-flash`). `predictions` scores an offline JSONL of detector boxes (`--predictions <file> --pred-model <model> [--pred-threshold <score>]`), for example from an object detector you run yourself. All engines use the same scoring, baselines and variants.
* `--annotate`: Write `annotated_<name>.svg` overlays with the argmax (dashed) and expectation (solid) boxes.
* `--simulate`: Offline harness check with synthetic slot distributions (no GPU).
* `-o`, `--output string`: Output JSON receipt path.

```bash
# 1. Every probe variant, 3 repeats
./bin/dgem bench-bbox --vertex-url <ENDPOINT_ID> --gcp-auth --variants all --repeat 3 -o receipt.json

# 2. Run a custom directory of images + index.txt prompts
./bin/dgem bench-bbox -u "<CLOUD_RUN_URL>/v1" --gcp-auth \
  --dir ./tmp/dgem-bounding-boxes --annotate \
  -o ./tmp/dgem-bounding-boxes/bbox_cloudrun.json
```


### `dgem bench-jev`
Runs [JevBench](https://github.com/fstandhartinger/jevbench), a public MIT-licensed zero-shot classification
benchmark (231 public items). The items are not in this repository: fetch and SHA-256-verify them first with
`--sync`, which downloads the release pinned in `benchmarks/jevbench/manifest.lock.json` (pass `--ref <tag>` to move
the pin); `--check-upstream` compares the pin with upstream.
Other flags: `--scoring v1.3.1|v1.4|both`, `--from-receipt <file> --auto-temperature` (offline temperature
calibration, fitted by 5-fold cross-validation; `--auto-temperature-in-sample` for the old in-sample fit), and
`--tier` / `--family` / `--topic` filters. Score items are sent as ordinal `score` questions with their rubric levels;
`--flatten-score` reproduces receipts made before 2026-10-04, which sent them as a choice over bare level indices.

```bash
./bin/dgem bench-jev --sync
./bin/dgem bench-jev --vertex-url <ENDPOINT_ID> --gcp-auth --http-retries 3 -w 4 -o jevbench.json
```

### `dgem bench-bbox-judge`
Validates a Gemini 3.x model as a **visual judge** of bounding boxes. It draws each proposed box on the image and asks the judge to grade every edge (`correct`, `too_far_in` or `too_far_out`) and the box overall. Verdicts are scored against ground truth:
- `--calibrate`: ground-truth boxes with one edge shifted in or out by known amounts (`--shifts 5,10,20`), plus the exact boxes.
- `--receipt <bench-bbox receipt>`: the boxes a localizer actually produced.

It reports three-class edge accuracy and κ, recall and specificity for wrong edges (`--tol`, `--off`), how often each error size is flagged (the judge's resolution), and box-level agreement. `--rescore <judge receipt>` recomputes the summaries without new Gemini calls. Ground truth comes from `-d/--dataset`
(default `benchmarks/bbox_suite.jsonl`); pass your own suite in the same format to judge other images.

```bash
./bin/dgem bench-bbox-judge --calibrate --receipt receipt.json --judge-model gemini-3.8-flash -o judge.json
```

### `dgem bench-vision`
Scores **multi-aspect image decisions**. One template (default `templates/multimodal/vision_aspects.json.tmpl`) asks several questions about each image in one pass: presence, 3×3 grid cell, element count, spatial relation, image quality and occlusion. The manifest (default `benchmarks/bbox_sweep.jsonl`) carries ground-truth `aspects` per item. Per aspect it reports:
- accuracy with a bootstrap interval;
- the majority-class baseline and the paired lift over it;
- the predicted-label shares;
- hesitation→error AUROC.

`--engine gemini` puts the same questions to Gemini 3.x. `--variant blank` measures the prompt prior. `--repeat N` measures run-to-run stability. `--only-scored` asks only the questions each item has ground truth for. `--cascade-threshold <nats>` (with `--cascade-model`) runs the live Stage-2 cascade: slots at or above the threshold go to Gemini together with the image. It reports dgem-only vs cascaded accuracy, the share of requests that called Gemini, and latency.

```bash
./bin/dgem bench-vision --vertex-url <ENDPOINT_ID> --gcp-auth --repeat 2 -o vision.json
./bin/dgem bench-vision --engine gemini --gemini-model gemini-3.8-flash -o vision_gemini.json
```

### `dgem bench-guided`
A harness for the guided locate pipeline (`dgem locate`): it compares Gemini 3.x boxes and masks with and without
a dgem pass first (skip when confidently absent, crop to dgem's cell, pass the cell as a hint) at different thinking
levels. Its dataset is not shipped: pass items with ground-truth boxes (`gt_box_continuous`) and presence labels
(`aspects.present`) with `-d/--dataset` (default path `third_party/bbox_real/bbox_real_aspects.jsonl`, which
`scripts/fetch_bbox_real.py` and `scripts/build_bbox_real_aspects.py` can build from public datasets). Choose
conditions with `--conditions` / `--neg-conditions` and the model with `--gemini-model` (default `gemini-3.8-flash`).

```bash
./bin/dgem bench-guided --vertex-url <ENDPOINT_ID> --gcp-auth -d my_items.jsonl -o guided.json
```

### `dgem ask`
Executes standard generative completions with optional `<|think|>` mode.

```bash
./bin/dgem ask "Explain discrete block diffusion in two sentences."
./bin/dgem ask --think "Verify whether 2015 + 4 precedes 2018."
```


### `dgem template`
Manages and inspects Go `.json.tmpl` policy templates locally without calling a GPU endpoint.

```bash
# List available templates across templates/ and templates/calibration/
./bin/dgem template list

# Render and inspect compiled JSON schema locally
./bin/dgem template render -t templates/calibration/chaos_nli.json.tmpl \
  -v premise="All four categories reached 50 to 75 percent of the cap." \
  -v hypothesis="Every category met the cap."
```
