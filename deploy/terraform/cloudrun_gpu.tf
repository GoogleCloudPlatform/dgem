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

resource "google_cloud_run_v2_service" "dgemma" {
  provider            = google-beta
  count               = var.enable_cloudrun_gpu ? 1 : 0
  project             = var.project_id
  name                = var.cloudrun_gpu_service_name
  location            = var.region
  launch_stage        = "BETA"
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  scaling {
    min_instance_count = var.cloudrun_gpu_min_instances
    max_instance_count = var.cloudrun_gpu_max_instances
  }

  template {
    service_account                  = google_service_account.gpu.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 32
    timeout                          = "600s"
    gpu_zonal_redundancy_disabled    = true

    node_selector {
      accelerator = var.cloudrun_gpu_type
    }

    dynamic "vpc_access" {
      for_each = var.enable_direct_vpc_egress ? [1] : []
      content {
        egress = "ALL_TRAFFIC"
        network_interfaces {
          network    = var.cloudrun_vpc_network
          subnetwork = var.cloudrun_vpc_subnet
        }
      }
    }

    volumes {
      name = "weights"
      gcs {
        bucket        = var.create_weights_bucket ? google_storage_bucket.weights[0].name : local.weights_bucket_name
        read_only     = true
        mount_options = ["enable-buffered-read=true"]
      }
    }

    containers {
      image = local.cloudrun_gpu_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu              = local.gpu_cpu
          memory           = local.gpu_memory
          "nvidia.com/gpu" = "1"
        }
        cpu_idle          = false
        startup_cpu_boost = true
      }

      volume_mounts {
        name       = "weights"
        mount_path = "/mnt/gcs"
      }

      startup_probe {
        initial_delay_seconds = 10
        period_seconds        = 5
        timeout_seconds       = 4
        failure_threshold     = 180
        http_get {
          path = "/ready"
          port = 8080
        }
      }

      env {
        name  = "MODEL"
        value = "/mnt/gcs/dgemma"
      }
      env {
        name  = "CANVAS"
        value = local.gpu_canvas_len
      }
      env {
        name  = "MAX_SEQS"
        value = "32"
      }
      env {
        name  = "MAX_MODEL_LEN"
        value = local.gpu_max_len
      }
      env {
        name  = "GPU_UTIL"
        value = "0.40"
      }
      env {
        name  = "KV_CACHE_GB"
        value = "2"
      }
      env {
        name  = "ATTN"
        value = "TRITON_ATTN"
      }
      env {
        name  = "ENFORCE_EAGER"
        value = "1"
      }
      env {
        name  = "DISABLE_MM"
        value = local.gpu_disable_mm
      }
      env {
        name  = "TORCH_COMPILE_DISABLE"
        value = "1"
      }
      env {
        name  = "VLLM_WORKER_MULTIPROC_METHOD"
        value = "fork"
      }
      env {
        name  = "CUDA_MODULE_LOADING"
        value = "LAZY"
      }
      env {
        name  = "COPY_TO_SHM"
        value = local.gpu_copy_to_shm
      }
      env {
        name  = "DEFAULT_SAMPLES"
        value = "1"
      }
      env {
        name  = "MAX_INFLIGHT"
        value = "8"
      }

      dynamic "env" {
        for_each = length(trimspace(var.hf_token_secret_id)) > 0 ? [trimspace(var.hf_token_secret_id)] : []
        content {
          name = "HF_TOKEN"
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
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
    google_storage_bucket_iam_member.gpu_weights_viewer,
    google_secret_manager_secret_iam_member.gpu_hf_token_accessor,
    google_secret_manager_secret_iam_member.gpu_server_key_accessor,
  ]
}
