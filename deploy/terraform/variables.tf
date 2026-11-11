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

variable "project_id" {
  description = "Target Google Cloud project ID."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) > 0 && var.project_id != "<PROJECT>"
    error_message = "project_id must be a real Google Cloud project ID (not empty or <PROJECT>)."
  }
}

variable "project_number" {
  description = "Optional Google Cloud project number override; defaults to data.google_project.current.number."
  type        = string
  default     = ""
}

variable "region" {
  description = "Google Cloud region for Cloud Run, Artifact Registry, Cloud Storage, and Vertex AI."
  type        = string
  default     = "us-central1"

  validation {
    condition     = can(regex("^[a-z]+-[a-z]+[0-9]+$", var.region))
    error_message = "region must be a valid Google Cloud region name such as us-central1."
  }
}

variable "allow_member" {
  description = "Authorized IAM principal granted Cloud Run invoker access (for example group:team@example.com or user:dana@example.com)."
  type        = string

  validation {
    condition = (
      can(regex("^(user|group|serviceAccount|domain):[^\\s]+$", var.allow_member)) &&
      var.allow_member != "allUsers" &&
      var.allow_member != "allAuthenticatedUsers"
    )
    error_message = "allow_member must start with user:, group:, serviceAccount:, or domain: (public principals are forbidden)."
  }
}

variable "enable_project_apis" {
  description = "Whether to enable required Google Cloud APIs on the target project."
  type        = bool
  default     = true
}

variable "create_artifact_registry" {
  description = "Whether to create the Artifact Registry Docker repository for dgemma and dgemma-gateway images."
  type        = bool
  default     = true
}

variable "artifact_registry_repository_id" {
  description = "Artifact Registry Docker repository ID."
  type        = string
  default     = "dgem"
}

variable "create_weights_bucket" {
  description = "Whether to create the Cloud Storage bucket holding staged DiffusionGemma weights."
  type        = bool
  default     = true
}

variable "weights_bucket_name" {
  description = "Cloud Storage bucket name for model weights; defaults to dgem-weights-<project_id> when empty."
  type        = string
  default     = ""
}

variable "enable_cloudrun_gpu" {
  description = "Whether to deploy the Cloud Run v2 GPU serving service (dgemma)."
  type        = bool
  default     = true
}

variable "cloudrun_gpu_service_name" {
  description = "Cloud Run v2 GPU service name."
  type        = string
  default     = "dgemma"
}

variable "cloudrun_gpu_type" {
  description = "Cloud Run GPU accelerator tier (nvidia-rtx-pro-6000 with SigLIP vision or nvidia-l4 text-only)."
  type        = string
  default     = "nvidia-rtx-pro-6000"

  validation {
    condition     = contains(["nvidia-rtx-pro-6000", "nvidia-l4"], var.cloudrun_gpu_type)
    error_message = "cloudrun_gpu_type must be either nvidia-rtx-pro-6000 or nvidia-l4."
  }
}

variable "cloudrun_gpu_image" {
  description = "Container image URI for the Cloud Run GPU service; defaults to <region>-docker.pkg.dev/<project_id>/<repo>/dgemma:latest."
  type        = string
  default     = ""
}

variable "cloudrun_gpu_min_instances" {
  description = "Minimum warm GPU instances on Cloud Run (0 enables scale-to-zero)."
  type        = number
  default     = 0
}

variable "cloudrun_gpu_max_instances" {
  description = "Maximum GPU instances on Cloud Run."
  type        = number
  default     = 1
}

variable "disable_multimodal" {
  description = "Override DISABLE_MM on the GPU serving container (defaults to false on RTX PRO 6000 and true on L4)."
  type        = bool
  default     = false
}

variable "enable_direct_vpc_egress" {
  description = "Attach Direct VPC Egress (ALL_TRAFFIC) so Cloud Storage FUSE stages weights over Private Google Access at ~410 MiB/s."
  type        = bool
  default     = true
}

variable "cloudrun_vpc_network" {
  description = "VPC network name used when enable_direct_vpc_egress is true."
  type        = string
  default     = "default"
}

variable "cloudrun_vpc_subnet" {
  description = "VPC subnetwork name (with Private Google Access enabled) used when enable_direct_vpc_egress is true."
  type        = string
  default     = "default"
}

variable "hf_token_secret_id" {
  description = "Optional Secret Manager secret ID holding HF_TOKEN (for example dgemma-hf-token); omitted when empty."
  type        = string
  default     = ""
}

variable "server_key_secret_id" {
  description = "Optional Secret Manager secret ID holding X-DGem-Key (for example dgem-server-key) shared between gateway and GPU service."
  type        = string
  default     = ""
}

variable "enable_gateway" {
  description = "Whether to deploy the Cloud Run HTTP API, MCP, and Web Studio gateway (dgemma-gateway)."
  type        = bool
  default     = true
}

variable "gateway_service_name" {
  description = "Cloud Run v2 service name for the dgem HTTP gateway."
  type        = string
  default     = "dgemma-gateway"
}

variable "gateway_image" {
  description = "Container image URI for dgemma-gateway; defaults to <region>-docker.pkg.dev/<project_id>/<repo>/dgemma-gateway:latest."
  type        = string
  default     = ""
}

variable "gateway_min_instances" {
  description = "Minimum instances for dgemma-gateway."
  type        = number
  default     = 1
}

variable "gateway_max_instances" {
  description = "Maximum instances for dgemma-gateway."
  type        = number
  default     = 10
}

variable "upstream_dgemma_url_override" {
  description = "Optional upstream /v1 URL when enable_cloudrun_gpu is false."
  type        = string
  default     = ""
}

variable "default_backend" {
  description = "Default gateway backend selection policy (vertex_first, vertex, or cloudrun)."
  type        = string
  default     = "vertex_first"

  validation {
    condition     = contains(["vertex_first", "vertex", "cloudrun"], var.default_backend)
    error_message = "default_backend must be one of vertex_first, vertex, or cloudrun."
  }
}

variable "gpu_idle_ttl" {
  description = "Idle TTL reported by the gateway for warm GPU lifecycle management."
  type        = string
  default     = "3h"
}

variable "verify_iap" {
  description = "Require IAP JWT signature and audience verification on every gateway request."
  type        = bool
  default     = true
}

variable "iap_audience_override" {
  description = "Custom IAP JWT audience string (for example a global BackendService audience); auto-computed for Cloud Run Direct IAP when empty."
  type        = string
  default     = ""
}

variable "enable_vertex_endpoint" {
  description = "Whether to provision a dedicated Vertex AI Endpoint shell (google_vertex_ai_endpoint) in the target region."
  type        = bool
  default     = false
}

variable "vertex_endpoint_name" {
  description = "Resource name and display name for the dedicated Vertex AI Endpoint."
  type        = string
  default     = "dgemma-dedicated"
}

variable "vertex_endpoint_url" {
  description = "Optional Vertex AI Endpoint ID or full /invoke URL wired into gateway DGEM_VERTEX_URL; uses google_vertex_ai_endpoint.dedicated[0].name when empty and enable_vertex_endpoint is true."
  type        = string
  default     = ""
}

variable "enable_monitoring" {
  description = "Whether to provision the six Cloud Logging decision metrics and the Cloud Monitoring operational dashboard."
  type        = bool
  default     = true
}

locals {
  resolved_project_number = (
    length(trimspace(var.project_number)) > 0
    ? trimspace(var.project_number)
    : tostring(data.google_project.current.number)
  )

  weights_bucket_name = (
    length(trimspace(var.weights_bucket_name)) > 0
    ? trimspace(var.weights_bucket_name)
    : "dgem-weights-${var.project_id}"
  )

  cloudrun_gpu_image = (
    length(trimspace(var.cloudrun_gpu_image)) > 0
    ? trimspace(var.cloudrun_gpu_image)
    : "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repository_id}/dgemma:latest"
  )

  gateway_image = (
    length(trimspace(var.gateway_image)) > 0
    ? trimspace(var.gateway_image)
    : "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repository_id}/dgemma-gateway:latest"
  )

  is_rtx_pro_6000 = var.cloudrun_gpu_type == "nvidia-rtx-pro-6000"
  gpu_cpu         = local.is_rtx_pro_6000 ? "20" : "8"
  gpu_memory      = local.is_rtx_pro_6000 ? "80Gi" : "32Gi"
  gpu_copy_to_shm = local.is_rtx_pro_6000 ? "1" : "0"
  gpu_canvas_len  = local.is_rtx_pro_6000 ? "128" : "32"
  gpu_max_len     = local.is_rtx_pro_6000 ? "8192" : "32768"
  gpu_disable_mm  = (var.disable_multimodal || !local.is_rtx_pro_6000) ? "1" : "0"

  resolved_iap_audience = (
    length(trimspace(var.iap_audience_override)) > 0
    ? trimspace(var.iap_audience_override)
    : "/projects/${local.resolved_project_number}/locations/${var.region}/services/${var.gateway_service_name}"
  )

  resolved_upstream_dgemma_url = (
    length(trimspace(var.upstream_dgemma_url_override)) > 0
    ? trimspace(var.upstream_dgemma_url_override)
    : (var.enable_cloudrun_gpu ? "${google_cloud_run_v2_service.dgemma[0].uri}/v1" : "")
  )

  resolved_vertex_url = (
    length(trimspace(var.vertex_endpoint_url)) > 0
    ? trimspace(var.vertex_endpoint_url)
    : (var.enable_vertex_endpoint ? google_vertex_ai_endpoint.dedicated[0].name : "")
  )
}
