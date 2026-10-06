# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: help build run test fmt clean setup download serve stop gateway-up gateway-down local-up local-down local-status release bench install docs-build docs-dev cloudrun-deploy gce-deploy gce-teardown check-public docs-sync-check

.DEFAULT_GOAL := help

# Version and build metadata
# Build version: the release tag (v0.1.0) on a release commit, v0.1.0-N-g<sha> after it (git describe).
# Releases are git tags: see `make release`.
BUILD_VERSION ?= $(shell git describe --tags --match 'v*' --always --dirty 2>/dev/null || echo "dev")
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "none")
DATE    ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -X github.com/GoogleCloudPlatform/dgem/cmd.Version=$(BUILD_VERSION) -X github.com/GoogleCloudPlatform/dgem/cmd.Commit=$(COMMIT) -X github.com/GoogleCloudPlatform/dgem/cmd.Date=$(DATE)

help: ## Show this help message
	@echo "dgem - DiffusionGemma Assistant & Structured Decision Engine"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

build: ## Build the dgem binary into bin/
	@mkdir -p bin
	go build -ldflags="$(LDFLAGS)" -o bin/dgem .

run: build ## Build and run dgem
	./bin/dgem

test: ## Run Go unit tests
	go test -v ./pkg/...

fmt: ## Format Go source code
	go fmt ./...

clean: ## Remove compiled binaries and temporary artifacts
	rm -rf bin/ dgem diffgemma.pid dgem-gateway.pid

setup: ## Verify prerequisites and install diffgemma engine
	./scripts/setup_diffgemma.sh

download: ## Download the 4-bit DiffusionGemma model pack (mmastrac/diffgemma-26b-a4b-it-q4)
	./scripts/download_model.sh

serve: ## Start the diffgemma server in background (32k context on 127.0.0.1:8080)
	./scripts/serve.sh

stop: ## Stop the background diffgemma server
	./scripts/stop_server.sh

gateway-up: build ## Start the dgem gateway & Decision Studio in background (:8090)
	./scripts/serve_gateway.sh

gateway-down: ## Stop the background dgem gateway
	./scripts/stop_gateway.sh

local-up: serve gateway-up ## Start full local stack: diffgemma engine (:8080) + dgem gateway (:8090)

local-down: gateway-down stop ## Stop full local stack: dgem gateway (:8090) + diffgemma engine (:8080)

local-status: ## Check running status of local diffgemma (:8080) and dgem gateway (:8090)
	@echo "=== Local DiffusionGemma Services Status ==="
	@if [ -f diffgemma.pid ] && kill -0 $$(cat diffgemma.pid) 2>/dev/null; then \
		echo "  • diffgemma engine:  RUNNING (PID $$(cat diffgemma.pid), port 8080)"; \
	elif lsof -i :8080 >/dev/null 2>&1; then \
		echo "  • diffgemma engine:  LISTENING (port 8080, unmanaged PID $$(lsof -t -i :8080 | tr '\n' ' '))"; \
	else \
		echo "  • diffgemma engine:  STOPPED"; \
	fi
	@if [ -f dgem-gateway.pid ] && kill -0 $$(cat dgem-gateway.pid) 2>/dev/null; then \
		echo "  • dgem gateway:      RUNNING (PID $$(cat dgem-gateway.pid), port 8090)"; \
	elif lsof -i :8090 >/dev/null 2>&1; then \
		echo "  • dgem gateway:      LISTENING (port 8090, unmanaged PID $$(lsof -t -i :8090 | tr '\n' ' '))"; \
	else \
		echo "  • dgem gateway:      STOPPED"; \
	fi
	@echo ""
	@if curl -s -m 1 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then \
		echo "  ✓ diffgemma API:     http://127.0.0.1:8080/v1 (ready)"; \
	else \
		echo "  ✗ diffgemma API:     offline"; \
	fi
	@if curl -s -m 1 http://127.0.0.1:8090/health >/dev/null 2>&1; then \
		echo "  ✓ Decision Studio:   http://localhost:8090 (ready)"; \
		echo "  ✓ REST API:          http://localhost:8090/api/decide"; \
		echo "  ✓ MCP Streamable:    http://localhost:8090/mcp"; \
	else \
		echo "  ✗ Decision Studio:   offline"; \
	fi

bench: build ## Run the local slot-readout vs autoregressive benchmark suite
	./bin/dgem bench

bench-calibration: build ## Run the public dataset calibration, guardrail, and human-entropy suite
	./bin/dgem bench-calibration

docs-build: ## Build the Astro Starlight documentation site
	pnpm run --dir docs-site build

docs-dev: ## Launch local development server for the documentation site
	pnpm run --dir docs-site dev

studio-build: ## Build the Lit + Vite Decision Studio WebComponents bundle (studio/dist)
	npm --prefix studio run build

studio-dev: ## Launch Vite HMR dev server for Decision Studio (proxies /api to 127.0.0.1:8080)
	npm --prefix studio run dev

cloudrun-build: ## Build and push the self-contained Cloud Run container image to Artifact Registry
	./scripts/build_cloudrun_image.sh

release: ## Tag a release: make release VERSION=v0.4.0 (clean main pushed to origin, tests, CHANGELOG entry, git tag)
	@echo "$(VERSION)" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$$' || { echo "usage: make release VERSION=vMAJOR.MINOR.PATCH"; exit 1; }
	@test -z "$$(git status --porcelain --untracked-files=no)" || { echo "working tree not clean"; exit 1; }
	@test "$$(git rev-parse --abbrev-ref HEAD)" = main || { echo "releases are cut from main"; exit 1; }
	@test "$$(git rev-parse HEAD)" = "$$(git rev-parse @{u} 2>/dev/null)" || { echo "push main first: the release tag must point at a commit already on origin"; exit 1; }
	@grep -q "^## $(VERSION)" CHANGELOG.md || { echo "add a '## $(VERSION)' section to CHANGELOG.md first"; exit 1; }
	@! git rev-parse -q --verify "refs/tags/$(VERSION)" >/dev/null || { echo "tag $(VERSION) already exists"; exit 1; }
	go test ./cmd/ ./pkg/...
	git tag -a $(VERSION) -m "dgem $(VERSION)"
	@echo "==> Tagged $(VERSION). Push it with 'git push origin $(VERSION)'. To serve it, build the image in your project: make cloudrun-build"

cloudrun-stage: ## Pre-stage model weights in GCS for Cloud Run GCS FUSE volume mounting
	./scripts/stage_model_gcs.sh

cloudrun-deploy: ## Deploy DiffusionGemma to Google Cloud Run with GPU (L4 or RTX Pro 6000)
	./scripts/deploy_cloudrun_vllm.sh

gateway-deploy: ## Deploy lightweight Go HTTP API & Web Studio Gateway (dgemma-gateway) to Cloud Run
	./scripts/deploy_cloudrun_gateway.sh

cloudrun-iam: ## Configure least-privilege Service Accounts (dgemma-gpu-sa, dgemma-gateway-sa) and Google Group IAM/IAP bindings
	./scripts/setup_cloudrun_iam.sh

cloudrun-dashboard: ## Provision Cloud Logging log-based metrics and Google Cloud Monitoring Operational Dashboard
	./scripts/setup_cloud_monitoring.sh

cloudrun-teardown: ## Delete Cloud Run dgemma GPU service to eliminate any cloud resource footprint
	gcloud run services delete dgemma --region=$${GCP_REGION:-us-central1} --quiet

gce-deploy: ## Deploy vLLM with DiffusionGemma structured reads to GCE with L4 GPU
	./scripts/deploy_gce_vllm.sh

gce-teardown: ## Delete GCE GPU benchmark instance and firewall to eliminate idle cost
	./scripts/teardown_gce_vllm.sh

vertex-deploy: ## Deploy DiffusionGemma to a Vertex AI Dedicated Endpoint with arbitrary custom routes (invokeRoutePrefix="/*")
	./scripts/deploy_vertex_endpoint.sh

vertex-teardown: ## Undeploy models and delete Vertex AI Dedicated Endpoint to eliminate idle GPU cost
	./scripts/teardown_vertex_endpoint.sh

install: ## Install dgem binary to GOBIN
	go install -ldflags="$(LDFLAGS)" .

check-public: ## Scan tracked files for identifiers you must not publish (regexes in the file named by DGEM_LEAK_PATTERNS_FILE)
	@echo "==> Scanning tracked files for leak patterns..."
	@python3 scripts/redact_receipt.py --check

docs-sync-check: ## Compare docs/ and docs-site/ page content (front matter, titles, and link targets ignored)
	@echo "==> Auditing documentation parity between docs/ and docs-site/..."
	@python3 scripts/docs_sync_check.py
