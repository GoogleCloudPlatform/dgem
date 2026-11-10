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

# Provisions the dedicated DNS Vertex AI Endpoint shell (`*.prediction.vertexai.goog`).
# Attaching a DeployedModel with `invokeRoutePrefix = "/*"` uses the Vertex v1beta1
# `:deployModel` API via `scripts/deploy_vertex_endpoint.sh` (see README.md).
resource "google_vertex_ai_endpoint" "dedicated" {
  count                      = var.enable_vertex_endpoint ? 1 : 0
  project                    = var.project_id
  name                       = var.vertex_endpoint_name
  display_name               = var.vertex_endpoint_name
  description                = "Dedicated DNS Vertex AI Endpoint for DiffusionGemma single-pass decision serving"
  location                   = var.region
  dedicated_endpoint_enabled = true

  depends_on = [google_project_service.required]
}
