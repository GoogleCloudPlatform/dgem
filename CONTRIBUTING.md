# How to contribute

To contribute code, decision templates, benchmarks, or documentation to `dgem`, follow the setup and verification steps in this guide.

## Before you begin

### Sign the Contributor License Agreement

Contributions to this project require a [Contributor License Agreement](https://cla.developers.google.com/about) (CLA). You (or your employer) retain the copyright to your contribution, and the CLA grants permission to use and redistribute your contributions as part of the project.

If you or your employer already signed the Google CLA for another project, you do not need to sign it again. Visit [https://cla.developers.google.com/](https://cla.developers.google.com/) to check your agreements or sign a new one.

### Review the community guidelines

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## Local development and testing

### Prerequisites

- **Go 1.24 or later** for the `dgem` CLI, gateway, MCP server, and `examples/` module.
- **Python 3.12 or later** for serving-image unit tests, the regression matrix (`scripts/bench_matrix.py`), and documentation sync scripts.
- **Node.js 22 and `pnpm` 10** for Decision Studio (`studio/`) and the Starlight documentation site (`docs-site/`).

### Build and run unit tests

Run the offline verification suite from the repository root before opening a pull request. Unset `DGEM_VERTEX_URL` in your shell first so unit tests run hermetically:

```bash
unset DGEM_VERTEX_URL
make build
go vet ./...
go test -race ./...
go -C examples vet ./...
go -C examples test -race ./...
```

Run the Python unit tests for the regression matrix, serving container, documentation link checker, and safety gate:

```bash
python3 -m unittest scripts/matrix/test_matrix.py
python3 -m unittest -v deploy/cloudrun/server/test_server_key.py \
  deploy/cloudrun/server/test_media_fields.py \
  deploy/cloudrun/server/test_upstream_errors.py \
  deploy/cloudrun/server/test_predict_route.py \
  deploy/cloudrun/server/test_label_sets.py
python3 -m unittest -v scripts/test_docs_link_check.py
python3 -m unittest -v examples/agy-safety-gate/test_dgem_gate.py
```

### Decision Studio (`studio/`)

When you add or update a decision option in `cmd/decide.go`, keep all four interaction surfaces aligned: the CLI (`cmd/decide.go`), the HTTP gateway (`cmd/serve.go`), the MCP server (`cmd/mcp.go`), and Decision Studio (`studio/src/dgem-studio.ts`).

After editing files under `studio/src/`, verify the TypeScript and Vite bundle, then clean untracked build artifacts before committing:

```bash
npm --prefix studio ci
npm --prefix studio run build
go build -o bin/dgem .
rm -rf studio/dist/* && touch studio/dist/.gitkeep
```

### Serving container patch (`deploy/cloudrun/server/`)

`deploy/cloudrun/server/structured_server.py` applies `dgem.patch` cleanly on top of `structured_server.upstream.py`. After editing `structured_server.py`, regenerate and verify the patch:

```bash
./scripts/regen_serving_patch.sh
```

## Update documentation

Source documentation lives in `docs/` and mirrors to the Astro Starlight site under `docs-site/`. After editing a page in `docs/`, sync it and verify parity, links, heading anchors, and the site build:

```bash
python3 scripts/docs_sync_page.py <relative-page-under-docs>
make docs-sync-check
python3 scripts/docs_link_check.py
pnpm --dir docs-site run build
```

Replace `<relative-page-under-docs>` with the path relative to `docs/`, such as `glossary.md` or `deploy/batch.md`.

Use standard placeholders in documentation and examples (`<PROJECT>`, `<PROJECT_NUMBER>`, `<REGION>`, `<ENDPOINT_ID>`, `<CLOUD_RUN_URL>`, and `<your-dgem-gateway>`). Never commit real project IDs, endpoint URLs, `.env` files, or credentials.

## Submit a pull request

All submissions, including submissions by project maintainers, require review through GitHub pull requests:

1. Create a feature branch from `main`.
2. Keep each pull request focused on a single concern.
3. Run the Go, Python, Decision Studio, and documentation checks that cover your changes, plus `make check-public` if you configure `DGEM_LEAK_PATTERNS_FILE`.
4. Open a pull request against `main` and summarize both what changed and how you verified it.
