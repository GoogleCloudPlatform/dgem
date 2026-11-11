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

resource "google_service_account" "gpu" {
  project      = var.project_id
  account_id   = "dgemma-gpu-sa"
  display_name = "DiffusionGemma GPU Engine SA (Read-Only GCS Weights)"
  description  = "Least-privilege identity for the dgemma Cloud Run GPU service; only reads staged weights."
}

resource "google_service_account" "gateway" {
  project      = var.project_id
  account_id   = "dgemma-gateway-sa"
  display_name = "DiffusionGemma HTTP Gateway SA (Invoker on dgemma GPU)"
  description  = "Least-privilege identity for dgemma-gateway; invokes dgemma Cloud Run and Vertex AI endpoints."
}

resource "google_storage_bucket_iam_member" "gpu_weights_viewer" {
  bucket = var.create_weights_bucket ? google_storage_bucket.weights[0].name : local.weights_bucket_name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gpu.email}"
}

resource "google_secret_manager_secret_iam_member" "gpu_hf_token_accessor" {
  count     = length(trimspace(var.hf_token_secret_id)) > 0 ? 1 : 0
  project   = var.project_id
  secret_id = trimspace(var.hf_token_secret_id)
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.gpu.email}"
}

resource "google_secret_manager_secret_iam_member" "gpu_server_key_accessor" {
  count     = length(trimspace(var.server_key_secret_id)) > 0 ? 1 : 0
  project   = var.project_id
  secret_id = trimspace(var.server_key_secret_id)
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.gpu.email}"
}

resource "google_secret_manager_secret_iam_member" "gateway_server_key_accessor" {
  count     = length(trimspace(var.server_key_secret_id)) > 0 ? 1 : 0
  project   = var.project_id
  secret_id = trimspace(var.server_key_secret_id)
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.gateway.email}"
}

resource "google_project_iam_member" "gateway_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.gateway.email}"
}

resource "google_project_iam_member" "gateway_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.gateway.email}"
}

resource "google_cloud_run_v2_service_iam_member" "gpu_gateway_invoker" {
  count    = var.enable_cloudrun_gpu ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.dgemma[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.gateway.email}"
}

resource "google_cloud_run_v2_service_iam_member" "gpu_allow_member_invoker" {
  count    = var.enable_cloudrun_gpu ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.dgemma[0].name
  role     = "roles/run.invoker"
  member   = var.allow_member
}

resource "google_cloud_run_v2_service_iam_member" "gateway_allow_member_invoker" {
  count    = var.enable_gateway ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.gateway[0].name
  role     = "roles/run.invoker"
  member   = var.allow_member
}
