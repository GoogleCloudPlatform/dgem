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

FROM node:22-alpine AS ui-builder
WORKDIR /ui
COPY studio/package.json studio/package-lock.json* studio/tsconfig.json studio/vite.config.ts studio/index.html ./
COPY studio/src ./src
RUN npm install --no-audit --no-fund && npm run build

FROM golang:1.26-alpine AS builder
ENV GOTOOLCHAIN=auto
ARG DGEM_VERSION=dev
ARG DGEM_REVISION=unknown
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
COPY --from=ui-builder /ui/dist ./studio/dist
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w -X github.com/GoogleCloudPlatform/dgem/cmd.Version=${DGEM_VERSION} -X github.com/GoogleCloudPlatform/dgem/cmd.Commit=${DGEM_REVISION}" -o /out/dgem .

FROM alpine:3.20
ARG DGEM_VERSION=dev
ARG DGEM_REVISION=unknown
LABEL org.opencontainers.image.title="dgem-gateway" \
      org.opencontainers.image.source="https://github.com/GoogleCloudPlatform/dgem" \
      org.opencontainers.image.version="${DGEM_VERSION}" \
      org.opencontainers.image.revision="${DGEM_REVISION}"
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /out/dgem /app/dgem
COPY templates /app/templates
ENV PORT=8080
ENV DGEM_GCP_AUTH=1
EXPOSE 8080
ENTRYPOINT ["/app/dgem", "serve", "--templates-dir", "/app/templates"]
