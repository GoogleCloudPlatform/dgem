# Changelog

Versioning: `vMAJOR.MINOR.PATCH`. **Major:** breaking API or response changes. **Minor:** vLLM runtime or model
changes, new features. **Patch:** fixes. Releases are git tags (`make release VERSION=vX.Y.Z`); build the serving
image for a release in your own project with `make cloudrun-build`.

## Unreleased

- **Serving image: optional service key (`X-DGem-Key`).** With `DGEM_SERVER_KEY` set (comma-separated to rotate),
  every `POST` needs a matching `X-DGem-Key` header; `GET /health` stays open. Unlike `API_KEY` it works on Vertex AI
  and Cloud Run, where `Authorization` carries the Google token. The `dgem` CLI, the gateway, `dgem systemone serve`,
  `scripts/matrix`, `probe.py`, `serving_speed.py` and `contract_diff.py` send the first entry of their own
  `DGEM_SERVER_KEY`; the gateway never forwards a caller's header. The Vertex AI, Cloud Run, gateway and probe deploy
  scripts set it from the Secret Manager secret `dgem-server-key` when it exists (`SERVER_KEY_SECRET` names another).
  Use it for any endpoint that more than your own services can reach, such as a Vertex AI dedicated endpoint
  ([Vertex AI](docs/deploy/vertex.md#1-prerequisites)).
- **Gateway: caller authentication.** `dgem serve --iap-audience` (`DGEM_IAP_AUDIENCE`) verifies the IAP JWT on every
  request (signature, audience, expiry) and takes the caller's identity from it, discarding identity headers sent by
  the client; `--api-keys` (`DGEM_GATEWAY_KEYS`, header `X-DGem-Api-Key`) serves self-hosted gateways without IAP.
  With either set, requests without valid credentials get 401 (`GET /health` stays open). `deploy_cloudrun_gateway.sh`
  sets the Cloud Run IAP audience by default (`VERIFY_IAP=0` to skip). The CLI and `dgem mcp --remote` send
  `DGEM_GATEWAY_KEY`; the gateway never forwards it upstream. Without either option the gateway warns at startup when
  it listens on a non-loopback address.
- **Image URLs are fetched by dgem, never by the serving image.** The gateway (`/api/decide`, MCP, `/api/locate`,
  `/v1/systemone` including `?raw=1`, and the `/v1/chat/completions` proxy) fetches http(s) image URLs with its
  public-address-only client and forwards data: URIs (unfetchable or non-image URLs: 422); the CLI fetches from the
  caller's network and does the same. **Behaviour change:** the serving image refuses http(s) image URLs with 400 by
  default (`DGEM_IMAGE_URLS=deny`; `allow` restores the old behaviour), because vLLM runs with network egress and a URL
  that passed the gateway's check could rebind or redirect to the metadata server or internal hosts.
- **Audio and video are refused with 400** by the serving image, the gateway and `dgem systemone serve` (`audio`,
  `video`, `input_audio`, `video_url` and similar fields or chat content parts), instead of being answered as if they
  were absent. Send video as sampled frames in `images` and audio as a transcript.
- **Requests the model server refuses are 4xx, not 500/502.** The serving image answers 422 (400 on the chat route)
  when vLLM refuses a request, such as an image it can't fetch or decode or a prompt over the context length; the
  gateway and `dgem systemone serve` pass such refusals on with the server's message. **Behaviour change:**
  `/api/decide` returns 422 instead of 502 for them and no longer retries them as cold starts; unreachable or failing
  servers stay 502. The gateway API reference documents the 400/401/422/502 meanings.
- **`/v1/systemone` adapter: score questions, one-option questions, `media`.**
  - `score` questions (`criteria` an array of levels) are read with the rest in batches of at most 8; answers follow
    the serving image's form (`score` = sum(i·p_i), probabilities keyed `"0"`..`"n-1"`, `legend`). Every answer now
    carries `confidence`.
  - A choice with one option or a score with one level is answered without a model call.
  - `media: [{mime_type, data}]` with `image/*` (data: URI or raw base64) is an alias for `images`; other types get 400.
  - **Behaviour change:** in adapter mode a body the adapter can't read (bad JSON, `questions` not an object, criteria
    of the wrong shape, unknown question type) gets 400 instead of silent pass-through; use `?raw=1` to send a body to
    the serving image unchanged. Score questions with more than 26 levels or repeated levels get 422.
- **Gateway logs say why a request failed, and who sent it.** Failed requests log the reason (`error`, up to 300
  characters) at WARNING (4xx) or ERROR (5xx) instead of INFO, on `/v1/systemone`, `/api/decide` and pass-through
  routes. The verified caller (`dgem.user`) is recorded on `/v1/systemone`, on `/mcp` (a new `dgem.gateway.mcp` span;
  tool arguments are never logged) and from the start of `/api/decide`. Adapter requests log their shape without
  content (question counts by type, max options, media entries, body size, forward passes). `/mcp` responses carry
  `X-Dgem-Trace-Id`.
- **Gateway: opt-in request/response body logging, off by default.** `DGEM_LOG_BODIES=responses|both` with a required
  `DGEM_LOG_BODIES_UNTIL` (RFC3339, at most 7 days ahead) and `DGEM_LOG_BODIES_USERS` (caller identities, or `*`),
  optional `DGEM_LOG_BODIES_SAMPLE`. Logs `/v1/systemone`, `/api/decide` and MCP `tools/call` bodies as
  `log_type: "dgem.bodies"` lines (96 KB cap per body; `data:` URIs and long base64 replaced by a digest, in JSON and
  non-JSON bodies; no headers or tokens). Invalid settings leave it off; startup and `/health` (`body_logging`) report
  its state. Setup, including a separate log bucket: [Observability](docs/operate/observability.md).
- **GCP tokens are cached only until they expire.** Access and ID tokens were cached for a flat 45 minutes, so an
  expired metadata-server token could be served: Vertex AI control-plane checks got 401 and a warm endpoint read as
  quiesced.
- **Gateway: `/v1/systemone` keeps images.** The adapter dropped the request's `images` and answered text-only with
  HTTP 200; it now sends them with every sub-request (batches and bracket rounds). Images must be data: URIs or
  public http(s) URLs (local paths are rejected with 400, in the gateway and in `dgem systemone serve`).
  `multipart/form-data` uploads failed with `400 invalid body` because the proxy forced `Content-Type:
  application/json`; it now forwards the caller's content type.
- **Gateway: `/v1/raw/chat/completions` removed (410 Gone).** dgem serves decisions; free-form generation had no
  callers. `/v1/chat/completions` on the gateway still makes decisions but is deprecated (`Deprecation` header); use
  `/api/decide` or `/v1/systemone`.
- **Serving image:**
  - Responses report `diagnostics.constrained`, which says how to read the per-question `entropy` and `label_mass`.
    With constrained reads (the default), `entropy` equals the entropy of the label probabilities and `label_mass` is
    always 1, so don't gate on it.
  - A read that needs more than 128 label tokens is refused with 422 instead of silently truncating vLLM's
    `logprob_token_ids`.
  - The vision tower is on by default (`DISABLE_MM=0`), as documented; set `DISABLE_MM=1` on an L4 with 32 GiB RAM.
  - `/v1/raw/chat/completions` answers `410 Gone`.
  - Experimental `seed_token: "random" | "fixed"` (per request, `DEFAULT_SEED_TOKEN`, or `--seed-token`): `fixed`
    seeds every answer position with the padding token instead of a seeded random token. Default `random` is unchanged.
- **Answers always carry an entropy.** The client fills each answer's `entropy` from its probabilities; the Studio
  could show 0% hesitation when the server didn't send one.
- **Opt-in normalized cascade gate.** `cascade_threshold_mode: "nats" | "normalized"` on `POST /api/decide` (or
  `X-DGem-Cascade-Threshold-Mode`), MCP and the Studio Batch Eval. `normalized` compares H / ln K (default
  threshold 0.16). The default stays `nats` (0.35). The two defaults are not equivalent operating points: 0.16
  normalized escalates more (31% vs 22% on a 341-decision test mix); fit the threshold on your own traffic.
- **SVG images are refused with a clear error** (CLI, gateway 400, `/v1/systemone`): the model server can't decode
  SVG and answered 502. Docs use new PNG fixtures (`fixtures/*.png`); the Studio picker offers raster formats only.
- **Templates keep the order of dict-form questions and options** (they were reordered randomly per call).
- **`dgem systemone serve`:** `--option-order alpha|source` (default `alpha`); server schema errors are passed on as
  422 with the Decision Index kit's marker instead of 500.
- **`bench-intents`:** kept intents above 26 go through another bracket round instead of a final the server rejects
  (full CLINC150).
- **Docs:** canvas size (128 rows as served, rounded to 16 per request), letter readout and the 26-option limit,
  stage scheduling, prompt-cache advice for `document_first`, and what constrained reads do.

## v0.4.0 — first public release

Earlier versions were developed internally; this is the first release in this repository.

- **Decision engine:** `dgem decide` compiles `.json.tmpl` decision policies into one-pass multi-question readouts
  (`boolean`, `choice`, `score`) with per-option probabilities, Shannon entropy and Hesitation %; conditional
  questions (`depends_on` / `ask_if`), per-policy `temperature`, and taxonomy-expansion suggestions.
- **Hesitation-gated cascade:** low-hesitation answers are returned directly; the rest go to a Stage 2 Gemini model
  (`gemini-3.8-flash` default, `gemini-3.5-flash-lite`) with a soft hint of dgem's answer by default
  (`stage2_prior: soft | full | none`).
- **Surfaces:** CLI, HTTP gateway (`dgem serve`: `/api/decide`, `/v1/systemone`, `/api/locate`), MCP server
  (`dgem mcp`, Streamable HTTP `/mcp`), and the embedded Decision Studio. Backend routing `vertex_first | vertex |
  cloudrun | local` with automatic failover.
- **Gateway hardening:** no CORS headers unless `--cors-origins` (`DGEM_CORS_ORIGINS`) lists exact origins (`*` is
  refused). Cross-site browser `POST`s and DNS-rebinding requests (a loopback connection carrying another `Host`) get
  `403`; `--allowed-hosts` (`DGEM_ALLOWED_HOSTS`) admits a reverse proxy on the same machine. Template names must be
  relative paths inside `--templates-dir`. The gateway fetches image URLs only from public addresses, checked when it
  connects and on every redirect; it does not read local image paths from requests, and an image URL that does not
  answer 2xx is an error. A per-request `vertex_url` is matched by project, region and endpoint ID, so another
  project's endpoint with the same ID no longer receives the gateway's token; regional
  `https://<REGION>-aiplatform.googleapis.com/...:rawPredict` URLs are accepted as well as dedicated endpoint hosts.
  The serving image sends `charset=utf-8` and `nosniff` on JSON responses.
- **Loopback bind by default:** `dgem serve` and `dgem systemone serve` listen on `127.0.0.1` unless `--host` says
  otherwise, so a laptop gateway is not reachable from the network by accident. On Cloud Run (`K_SERVICE`) and Vertex
  (`AIP_HTTP_PORT`) the default is `0.0.0.0`, and the gateway image and the serving entrypoint pass `--host 0.0.0.0`
  explicitly. To reach a gateway from another machine, pass `--host 0.0.0.0` and put it behind IAP/IAM.
- **Images:** approximate one-pass boxes, and guided locate (`dgem locate`, MCP `locate_object`): dgem answers
  presence and grid cell, then Gemini returns the box; optional SAM masks.
- **Serving image:** vLLM structured generation with the dgem server patch (`deploy/cloudrun/`): `document_first`
  prompt layout, yes/no isolation (`DEFAULT_ISOLATE=auto`), in-flight limiting, 8k context on RTX PRO 6000,
  idle-connection timeout. Deploy scripts for Vertex AI dedicated endpoints, Cloud Run GPU and GCE.
- **Evaluation:** `dgem bench-*` harnesses and the regression matrix (`scripts/bench_matrix.py`,
  `benchmarks/matrix/matrix_example.json`) to compare a candidate deployment with a baseline in one session.
