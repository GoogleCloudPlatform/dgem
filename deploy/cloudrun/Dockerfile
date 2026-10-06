# syntax=docker/dockerfile:1
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

# DiffusionGemma Production & Decision Index Container
# Built cleanly on official vLLM nightly pinned by digest (commit a9eafde59cb).
# Upstream PR #57250 and PR #58216 are natively included in the base image.

ARG BASE=vllm/vllm-openai:nightly-a9eafde59cbd55182dc2265cc398b7186f0a0eaa@sha256:896b7652343b6230cf788f4f1744563539f461f0454f30c5c7bb96ee9f8a34be

FROM ${BASE} AS base
ARG VLLM_COMMIT=a9eafde59cbd55182dc2265cc398b7186f0a0eaa
# Release version (git describe, e.g. v0.1.0) and commit, passed by deploy/cloudrun/cloudbuild.yaml.
ARG DGEM_VERSION=dev
ARG DGEM_REVISION=unknown
LABEL org.opencontainers.image.title="dgem" \
      org.opencontainers.image.description="DiffusionGemma decision-model serving (vLLM + dgem structured server)" \
      org.opencontainers.image.source="https://github.com/GoogleCloudPlatform/dgem" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${DGEM_VERSION}" \
      org.opencontainers.image.revision="${DGEM_REVISION}" \
      dev.dgem.vllm.commit="${VLLM_COMMIT}"
# Reported by /health so a running service identifies its release.
ENV DGEM_VERSION=${DGEM_VERSION} DGEM_REVISION=${DGEM_REVISION} DGEM_VLLM_COMMIT=${VLLM_COMMIT}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 1. Symlink CUDA headers where FlashInfer/CUTLASS JIT requires them
COPY patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN bash /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

# 2. Install compressed-tensors and huggingface_hub
RUN pip install --no-cache-dir compressed-tensors huggingface_hub

# 3. Bundle dgem binary
COPY dgem /usr/local/bin/dgem
RUN chmod +x /usr/local/bin/dgem

# 4. Bundle patched structured_server proxy and entrypoint
COPY server/structured_server.py /opt/dgemma/structured_server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /opt/dgemma/structured_server.py \
    && python3 -m py_compile /opt/dgemma/structured_server.py

EXPOSE 8080 8081 8000
ENTRYPOINT ["/entrypoint.sh"]
