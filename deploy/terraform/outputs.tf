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

output "weights_bucket_name" {
  description = "Cloud Storage bucket holding staged DiffusionGemma weights."
  value       = local.weights_bucket_name
}

output "gpu_service_account_email" {
  description = "Least-privilege service account email running the dgemma Cloud Run GPU service."
  value       = google_service_account.gpu.email
}

output "gateway_service_account_email" {
  description = "Least-privilege service account email running dgemma-gateway."
  value       = google_service_account.gateway.email
}

output "cloudrun_gpu_uri" {
  description = "HTTPS URL of the Cloud Run v2 GPU service (empty when enable_cloudrun_gpu is false)."
  value       = var.enable_cloudrun_gpu ? google_cloud_run_v2_service.dgemma[0].uri : ""
}

output "gateway_uri" {
  description = "HTTPS URL of the dgemma-gateway Cloud Run service (empty when enable_gateway is false)."
  value       = var.enable_gateway ? google_cloud_run_v2_service.gateway[0].uri : ""
}

output "iap_audience" {
  description = "Expected IAP JWT audience configured on dgemma-gateway."
  value       = var.verify_iap ? local.resolved_iap_audience : ""
}

output "vertex_endpoint_name" {
  description = "Numeric Vertex AI Endpoint ID when enable_vertex_endpoint is true."
  value       = var.enable_vertex_endpoint ? google_vertex_ai_endpoint.dedicated[0].name : ""
}

output "vertex_dedicated_endpoint_dns" {
  description = "Dedicated prediction DNS hostname when enable_vertex_endpoint is true."
  value       = var.enable_vertex_endpoint ? google_vertex_ai_endpoint.dedicated[0].dedicated_endpoint_dns : ""
}

output "monitoring_dashboard_id" {
  description = "Cloud Monitoring operational dashboard resource ID when enable_monitoring is true."
  value       = var.enable_monitoring ? google_monitoring_dashboard.operational[0].id : ""
}
