# dgem contributor and agent guidelines

Guidance for people and coding agents changing this repository. User documentation is in `docs/`
(published at https://googlecloudplatform.github.io/dgem/).

## Source of truth
This repository (module `github.com/GoogleCloudPlatform/dgem`) is where dgem is developed, released and built.
Earlier development happened in a previous repository, which is frozen and will be archived. Open issues and pull
requests here. Changes from the previous repository are brought over as rewritten ports (module path
`github.com/GoogleCloudPlatform/dgem`); nothing is ported back. Don't port its move notice (README callout, docs
banner, issue-template redirect, `go.mod` `Deprecated` line).

## Layout
- `cmd/`: CLI, HTTP gateway (`dgem serve`), MCP server (`dgem mcp`), Stage 2 cascade, bench harnesses.
- `pkg/client`: client for the serving backends (Vertex AI `/invoke/*`, Cloud Run / vLLM, local Metal `diffgemma`).
- `pkg/template`, `templates/`: `.json.tmpl` decision policies (Policy-as-Template).
- `pkg/decisionindex`: `/v1/systemone` adapter engine. `pkg/permutation`: option-order tools.
- `studio/`: Decision Studio (Lit + Vite), embedded into the binary from `studio/dist` via `go:embed`.
- `deploy/cloudrun/`: serving image (`Dockerfile`, `entrypoint.sh`, `server/structured_server.py`).
- `scripts/`: deploy, build, staging, monitoring and evaluation scripts; `scripts/bench_matrix.py` + `scripts/matrix/`.
- `docs/` (source) and `docs-site/` (Astro Starlight mirror).

## Build and test
```bash
make build                      # ./bin/dgem
go vet ./... && go test ./...   # unset DGEM_VERTEX_URL first if it is set in your shell
python3 -m unittest scripts/matrix/test_matrix.py
python3 -m unittest deploy/cloudrun/server/test_server_key.py deploy/cloudrun/server/test_media_fields.py \
  deploy/cloudrun/server/test_upstream_errors.py
```
After changing `studio/src/*`: `npm --prefix studio run build && go build -o bin/dgem .`, then
`rm -rf studio/dist/* && touch studio/dist/.gitkeep` before committing (build output is not tracked).

## Invariants
- **Four-surface parity.** A new decision option or post-processing pass in `dgem decide` (`cmd/decide.go`) is also
  exposed in `GatewayDecideRequest` / `GatewayDecideResponse` (`cmd/serve.go`), the MCP tool inputs (`cmd/mcp.go`),
  and the Studio (`studio/src/dgem-studio.ts`).
- **Tracing.** Anything that triggers another model pass or a Stage 2 call runs in its own OpenTelemetry child span
  under `dgem.gateway.decide` (for example `dgem.cascade.gemini`, `dgem.taxonomy.expand`).
- **Gateway callers and logs.** `callerAuthHandler` (`cmd/caller_auth.go`) is the only source of the caller identity
  (`dgem.user`); identity headers sent by clients are discarded. Gateway logs carry status, reason, caller and request
  shape, never request content: content goes only to the opt-in, time-limited body log (`cmd/body_log.go`), with
  media redacted. A new route or failure path records its status and reason with `failSpan`.
- **Unauthenticated replies reveal no configuration.** The gateway's 401 body is the same whatever is configured, and
  the open `GET /health` (on `dgem serve` and `dgem systemone serve`) reports liveness and version only (plus
  `body_logging.enabled` on the gateway). Authentication methods, the IAP audience, backends, upstream URLs and
  body-logging settings go to the startup log or to authenticated routes (`/api/backend-config`, `/api/status`).
- **Each credential goes only to its own service.** `X-DGem-Key` (`DGEM_SERVER_KEY`) goes to the serving image, and
  `X-DGem-Api-Key` (`DGEM_GATEWAY_KEY`) only to the gateway in `DGEM_GATEWAY_URL` (`client.GatewayKeyFor`) or to the
  `dgem mcp --remote` URL (`client.GatewayKeyForRemote`). HTTP clients that send either strip credentials on
  cross-origin redirects (`client.StripKeysOnCrossOriginRedirect` in Go, `_RedirectHandler` in `scripts/matrix/net.py`
  and `scripts/serving_speed.py`), and the gateway never forwards a caller's key.
- **Images reach the serving image as data: URIs.** The gateway and the CLI fetch http(s) image URLs themselves with
  the guarded client (public addresses only, checked on every connection and redirect); the serving image refuses
  URLs by default (`DGEM_IMAGE_URLS=deny`). Requests the model server refuses are 4xx (`decisionindex.ErrorStatus`),
  never 500/502.
- **Stage 2 models.** Only generally available Gemini models: `gemini-3.8-flash` (default) and
  `gemini-3.5-flash-lite`. `SanitizeCascadeModel` maps older or pre-GA names to these.
- **Backend telemetry stays separate.** Use `RecordVertexReadoutLatency` for Vertex AI, `RecordLocalReadoutLatency`
  for local `diffgemma`, and `MarkGPUWarm` only for Cloud Run, so backend status reports stay accurate.
- **MCP output schemas.** `StructuredDecisionResponse.UnmarshalJSON` initialises `Answers`,
  `Diagnostics.Questions` and `QuestionAnswer.Probabilities` to empty maps so MCP output validation never sees `null`.
- **Remote MCP proxy.** `runRemoteMCPProxy` handles each stdin request in its own goroutine with a mutex-guarded
  stdout writer, so long-lived streams never block other requests.
- **Choice limits.** The serving image maps `choice` options to letters `A`–`Z`: at most 26 options per question.
  Use a slice or two-stage routing for larger taxonomies (`pkg/decisionindex` brackets them automatically).
- **`depends_on` / `ask_if`** split questions into two sequential reads; boolean answers are `"yes"`/`"no"`, so
  `ask_if` uses `["yes"]`. Leave them out when one read matters.
- **Samples.** Templates default to `samples: 1`; use `4` only where agreement matters, via
  `{{ default 1 .samples | toJson }}` so callers can override.
- **Question ids are part of the prompt.** Use neutral, descriptive ids; never ids that hint the answer should
  differ (`mirror`, `reverse`, `opposite`, `alt`). Avoid shared option letters across questions.
- **Uncertainty in the UI** is shown as Hesitation % (normalized entropy; <16% clear, 16–50% somewhat unsure,
  >50% very unsure) via `studio/src/hesitation.ts`. `cascade_threshold` is raw nats. Raw restricted-softmax
  probabilities are an uncertainty signal; call probabilities "calibrated" only when fitted per policy on held-out data.
  Label simulated numbers as illustrative.

## Serving image
- `deploy/cloudrun/server/structured_server.py` is vLLM's upstream example (`structured_server.upstream.py`, kept
  byte-identical) plus `dgem.patch`. Edit `structured_server.py`, then run `scripts/regen_serving_patch.sh`: it
  regenerates the patch in GNU `diff -u` form and checks that it re-applies to the upstream file, as CI does (plain
  `diff -u` on macOS produces different hunks).
- Keep in the patch: `/health` merging `/tmp/dgemma/warmup_state.json`, HTTP/1.1 keep-alive with listen backlog 256,
  `MAX_INFLIGHT`, `DEFAULT_SAMPLES`, `DEFAULT_LAYOUT` (default `document_first`), `DEFAULT_ISOLATE` (default `auto`),
  the `/predict` / `/rawPredict` aliases, the `X-DGem-Key` check, model-server 4xx answered as 422 (400 on the chat
  route), the audio/video refusal (400), and `DGEM_IMAGE_URLS` (default `deny`).
- **Service key.** With `DGEM_SERVER_KEY` set (comma-separated to rotate), the serving image refuses every `POST`
  without a matching `X-DGem-Key` header; `GET /health` stays open. Callers send the first entry of their own
  `DGEM_SERVER_KEY`: `pkg/client`, the gateway proxy in `cmd/serve.go` (which never forwards a caller's header),
  `scripts/matrix/net.py` and `scripts/serving_speed.py` (also used by `probe.py` and `contract_diff.py`). A new
  caller of the serving image sends it too. The key is for services, never for people, and is never committed.
- Validate a serving change by running the new image next to the current one and comparing them with
  `scripts/bench_matrix.py run --tier T1 --target current=<url> --target new=<url> --baseline current`
  (T2 with `--confirm` for vLLM upgrades). Results are judged against the noise measured in the same session.
- Build images into your own Artifact Registry (`make cloudrun-build`). No images are published from this repo yet.

## Docs
- Edit `docs/`, then copy to the site with `python3 scripts/docs_sync_page.py <page>`; check with
  `make docs-sync-check`, `python3 scripts/docs_link_check.py` and `pnpm --dir docs-site run build`.
- When moving a page, add a redirect in `docs-site/astro.config.mjs`.
- How-to pages use placeholders only: `<PROJECT>`, `<PROJECT_NUMBER>`, `<REGION>`, `<ENDPOINT_ID>`,
  `<CLOUD_RUN_URL>`, `<your-dgem-gateway>`. Never commit project IDs, endpoint IDs or internal hostnames.
- Keep the Studio glossary (`studio/src/components/dgem-concept-visualizer.ts`, `dgem-about-modal.ts`),
  `docs-site/public/visualizer.html` and `docs/glossary.md` consistent.

## Datasets
Record the source and licence of every evaluation dataset in `benchmarks/README.md`. Any committed dataset derived from
third-party material must live in `third_party/<dataset>/` alongside its `LICENSE` file. Prefer a pinned fetch script
(with SHA-256 verification) over committing third-party data.

## Hygiene
- Never commit `.env` files, tokens, keys, `*.log` files or receipts containing endpoint URLs
  (`scripts/redact_receipt.py` can redact them).
- `make check-public` scans tracked files against a pattern file you provide (`DGEM_LEAK_PATTERNS_FILE`).
- Tear down GPU resources you create for testing (`make gce-teardown`, `make vertex-teardown`,
  `make cloudrun-teardown`).
