#!/usr/bin/env bash
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

set -euo pipefail

# Build and Push DiffusionGemma Cloud Run Image to Google Artifact Registry
# Uses Cloud Build for 100% cloud-native, reproducible container building.

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
REPOSITORY="${AR_REPOSITORY:-dgem}"
IMAGE_NAME="${IMAGE_NAME:-dgemma}"
TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: No GCP project detected. Set GCP_PROJECT=<project-id>."
  exit 1
fi

AR_TARGET="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo "================================================================================"
echo "  Building DiffusionGemma Cloud Run Image via Google Cloud Build"
echo "================================================================================"
echo "Project:    $PROJECT_ID"
echo "Region:     $REGION"
echo "Repository: $REPOSITORY"
echo "Target Tag: $AR_TARGET"
echo "Source:     deploy/cloudrun/"
echo "================================================================================"

# 1. Ensure Artifact Registry repository exists
if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "==> Creating Artifact Registry repository $REPOSITORY in $REGION..."
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    --description="DiffusionGemma Cloud Run container images"
fi

# 2. The Dockerfile bundles a static dgem binary (systemone adapter for ROLE=decision-index)
echo "==> Building static Linux dgem binary into deploy/cloudrun/dgem..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-X github.com/GoogleCloudPlatform/dgem/cmd.Version=$(git describe --tags --match 'v*' --always --dirty 2>/dev/null || echo dev) -X github.com/GoogleCloudPlatform/dgem/cmd.Commit=$(git rev-parse --short HEAD)" -o deploy/cloudrun/dgem .
trap 'rm -f deploy/cloudrun/dgem' EXIT

VERSION_STR="$(git describe --tags --match 'v*' --always --dirty 2>/dev/null || echo dev)"
echo "==> Submitting build to Google Cloud Build (version ${VERSION_STR})..."
gcloud builds submit deploy/cloudrun \
  --project="$PROJECT_ID" \
  --config=deploy/cloudrun/cloudbuild.yaml \
  --substitutions="_IMAGE=${AR_TARGET%:*},_TAG=${TAG},_VERSION=${VERSION_STR}"

echo ""
echo "================================================================================"
echo "  Container Image Ready!"
echo "================================================================================"
echo "Image URI: $AR_TARGET"
echo ""
echo "Deploy to Cloud Run:"
echo "  CLOUDRUN_IMAGE=\"$AR_TARGET\" make cloudrun-deploy"
echo "================================================================================"
