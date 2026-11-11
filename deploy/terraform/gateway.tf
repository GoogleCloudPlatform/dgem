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

resource "google_cloud_run_v2_service" "gateway" {
  count               = var.enable_gateway ? 1 : 0
  project             = var.project_id
  name                = var.gateway_service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  scaling {
    min_instance_count = var.gateway_min_instances
    max_instance_count = var.gateway_max_instances
  }

  template {
    service_account                  = google_service_account.gateway.email
    max_instance_request_concurrency = 80
    timeout                          = "600s"

    containers {
      image = local.gateway_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = false
      }

      env {
        name  = "DGEM_GCP_AUTH"
        value = "1"
      }
      env {
        name  = "DGEM_GPU_IDLE_TTL"
        value = var.gpu_idle_ttl
      }
      env {
        name  = "DGEM_DEFAULT_BACKEND"
        value = var.default_backend
      }
      env {
        name  = "DGEM_GCP_PROJECT_NUMBER"
        value = local.resolved_project_number
      }

      dynamic "env" {
        for_each = length(local.resolved_upstream_dgemma_url) > 0 ? [local.resolved_upstream_dgemma_url] : []
        content {
          name  = "UPSTREAM_DGEMMA_URL"
          value = env.value
        }
      }

      dynamic "env" {
        for_each = length(local.resolved_vertex_url) > 0 ? [local.resolved_vertex_url] : []
        content {
          name  = "DGEM_VERTEX_URL"
          value = env.value
        }
      }

      dynamic "env" {
        for_each = var.verify_iap ? [local.resolved_iap_audience] : []
        content {
          name  = "DGEM_IAP_AUDIENCE"
          value = env.value
        }
      }

      dynamic "env" {
        for_each = length(trimspace(var.server_key_secret_id)) > 0 ? [trimspace(var.server_key_secret_id)] : []
        content {
          name = "DGEM_SERVER_KEY"
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_project_iam_member.gateway_trace_agent,
    google_project_iam_member.gateway_vertex_user,
    google_secret_manager_secret_iam_member.gateway_server_key_accessor,
  ]
}
