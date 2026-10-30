---
title: "Security and Access"
description: "Who can reach the dgem gateway and the serving image, how image URLs are fetched, what the gateway logs, and how to report a vulnerability."
---

dgem runs two network services. The **gateway** (`dgem serve`) serves the HTTP API, MCP and the Studio to people and
agents. The **serving image** (the model server on Cloud Run or a Vertex AI endpoint) is called by services: the
gateway, `dgem systemone serve`, batch jobs and probes. This page lists the controls on each and links to the setup
details.

## Gateway

| Control | Setting | Default |
| :--- | :--- | :--- |
| Listen address | `--host` | `127.0.0.1`; `0.0.0.0` on Cloud Run and Vertex AI |
| Caller authentication with IAP | `--iap-audience` (`DGEM_IAP_AUDIENCE`) | off; `deploy_cloudrun_gateway.sh` sets it (`VERIFY_IAP=0` skips) |
| Caller authentication with API keys | `--api-keys` (`DGEM_GATEWAY_KEYS`), sent as `X-DGem-Api-Key` | off |
| Browser origins | `--cors-origins` (`DGEM_CORS_ORIGINS`) | none: cross-site `POST`s get `403`; `*` is refused |
| `Host` names on loopback | `--allowed-hosts` (`DGEM_ALLOWED_HOSTS`) | loopback names only; others get `403` (DNS rebinding) |
| Backends | `--backends` (`DGEM_BACKENDS`) | derived from the configured URLs |
| Vertex AI endpoints a request may select | `--allowed-vertex-endpoints` (`DGEM_ALLOWED_VERTEX_ENDPOINTS`) | the configured endpoint only |
| Admin actions (gateway-wide routing, endpoint deploy and teardown) | `--enable-admin-api` (`DGEM_ADMIN_API=1`) | off: `403` |

With IAP or API keys set, every request except the health checks needs valid credentials (`401` otherwise), and the
caller's identity comes only from the verified credential: identity headers sent by clients are discarded. Replies to
callers that haven't authenticated reveal nothing about the setup: every refusal is the same `401`, and the open
`GET /health` reports only liveness, the version and whether body logging is on. The accepted methods and the
body-logging settings are printed in the gateway's startup log. With neither set, the gateway warns at startup when
it listens on a non-loopback address; keep such a gateway behind IAP or Cloud Run IAM (`deploy_cloudrun_gateway.sh`
deploys with `--no-allow-unauthenticated` and grants `ALLOW_GROUP`). A browser can't send an API key, so the Studio
needs IAP or a loopback gateway without keys. Setup:
[Gateway and Routing](/dgem/deploy/gateway/#access-for-people-and-agents).

Each key goes only to its own service. The `dgem` CLI sends its API key (`DGEM_GATEWAY_KEY`) only to the gateway in
`DGEM_GATEWAY_URL` (same scheme, host and port), never to Vertex AI, Cloud Run or another URL it is pointed at;
`dgem mcp --remote` sends it to its remote. No dgem client carries an API key or a service key (`X-DGem-Key`) across
a redirect to another origin, and the gateway never forwards a caller's key.

Template names can't leave the templates directory: `/api/decide/{template}`, the `template` field and the MCP
`decide_policy` tool accept template IDs only (`400` otherwise).

## Serving image

- **Service key.** With `DGEM_SERVER_KEY` set (comma-separated to rotate), every `POST` needs a matching `X-DGem-Key`
  header; `GET /health` stays open. The Vertex AI, Cloud Run, gateway and probe deploy scripts set it from the Secret
  Manager secret `dgem-server-key` when it exists (`SERVER_KEY_SECRET` names another). Callers send the first entry
  of their own `DGEM_SERVER_KEY`; the gateway never forwards a caller's header. Use a key for any endpoint that more
  than your own services can reach, such as a Vertex AI dedicated endpoint
  ([Vertex AI prerequisites](/dgem/deploy/vertex/#1-prerequisites)). The key is for services, never for people.
- **Cloud Run IAM.** The Cloud Run deploy script uses `--no-allow-unauthenticated`, so callers also need
  `roles/run.invoker` on the service.
- **No URL fetching.** With `DGEM_IMAGE_URLS=deny` (the default) the serving image takes images only as data: URIs or
  multipart uploads and refuses http(s) URLs with `400`. vLLM runs with network egress, so a URL could resolve or
  redirect to the metadata server or internal hosts. `allow` restores URL fetching; don't use it on an image that can
  reach internal networks.

## Image URLs

The gateway and the `dgem` CLI fetch http(s) image URLs themselves and send the image bytes as data: URIs.

- The gateway (`/api/decide`, MCP, `/api/locate`, `/v1/systemone` including `?raw=1`, and the `/v1/chat/completions`
  proxy) fetches only from public addresses. Every connection is checked after DNS resolution, including at most 3
  redirects (http and https only), and proxy settings are ignored. Loopback, private, link-local (including the
  metadata server at `169.254.169.254`), CGNAT, multicast and reserved addresses are refused.
- Content must be a raster image (no SVG or HTML) of at most 20 MB. URLs that can't be fetched or aren't images get
  `422`; refused references, including local file paths, get `400`.
- The CLI fetches from your own network.

## Logs

- Gateway logs carry the caller (`dgem.user`), route, status, failure reason and request shape (question counts, image
  count, body size), never request or response content. Refused requests log at WARNING and server faults at ERROR
  ([Observability and Traces](/dgem/operate/observability/)).
- Request and response bodies are logged only when you turn on
  [body logging](/dgem/operate/observability/#4-request-and-response-body-logging-off-by-default): off by default, for named
  callers, until a time at most 7 days ahead, with images redacted. Route those entries to a separate log bucket
  first.
- Keep service and API keys in Secret Manager (the deploy scripts read `dgem-server-key` from it); never commit them
  or build them into images.

## Upgrading an existing deployment

When you move a gateway and serving image deployed from an earlier build to one with these controls:

1. **Deploy the gateway first.** The new serving image refuses image URLs (`DGEM_IMAGE_URLS=deny`), and an older
   gateway forwards them unchanged. The new gateway sends images as data: URIs, so deploy it before the serving image.
2. **Expect IAP verification after a gateway redeploy.** `deploy_cloudrun_gateway.sh` sets `DGEM_IAP_AUDIENCE`, so the
   redeployed gateway verifies the IAP JWT on every request (`IAP_AUDIENCE` names a load-balancer backend service
   instead; `VERIFY_IAP=0` skips verification). Deploy with `GATEWAY_TAG`, then check `GET /health` and one signed-in
   request on the tagged revision before moving traffic
   ([verify a new gateway revision](/dgem/deploy/gateway/#deploy-the-gateway-on-cloud-run)).
3. **Body logging stays off** unless you configure it.
4. **CLI callers with an API key** add `DGEM_GATEWAY_URL` (the gateway's URL) next to `DGEM_GATEWAY_KEY`: without it
   the CLI sends no key, and its error on the gateway's `401` says so. `dgem mcp --remote` needs no change.

## Reporting a vulnerability

Report security issues through Google's [vulnerability reporting process](https://g.co/vulnz), not in public issues.
The repository's [security policy](https://github.com/GoogleCloudPlatform/dgem/security/policy) describes how reports
are handled.
