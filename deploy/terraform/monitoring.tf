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

resource "google_logging_metric" "dgem_decisions_total" {
  count       = var.enable_monitoring ? 1 : 0
  project     = var.project_id
  name        = "dgem_decisions_total"
  description = "Total DiffusionGemma decision evaluations across Web Studio, MCP, and REST API"
  filter      = "resource.type=\"cloud_run_revision\" AND jsonPayload.span_name=\"dgem.gateway.decide\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    labels {
      key         = "surface"
      value_type  = "STRING"
      description = "Access surface (web_studio, mcp_agent, rest_api, openai_compat)"
    }
    labels {
      key         = "template"
      value_type  = "STRING"
      description = "Decision policy template name (.json.tmpl)"
    }
    labels {
      key         = "user"
      value_type  = "STRING"
      description = "Authenticated user email"
    }
    labels {
      key         = "multimodal"
      value_type  = "STRING"
      description = "Whether SigLIP vision input was used (true/false)"
    }
    labels {
      key         = "reads"
      value_type  = "STRING"
      description = "Forward passes (1 = single-pass O(1), 2+ = conditional DAG)"
    }
  }

  label_extractors = {
    surface    = "EXTRACT(jsonPayload.dgem_surface)"
    template   = "EXTRACT(jsonPayload.dgem_template)"
    user       = "EXTRACT(jsonPayload.dgem_user)"
    multimodal = "EXTRACT(jsonPayload.dgem_multimodal)"
    reads      = "EXTRACT(jsonPayload.dgem_gpu_reads)"
  }

  depends_on = [google_project_service.required]
}

resource "google_logging_metric" "dgem_epistemic_entropy" {
  count           = var.enable_monitoring ? 1 : 0
  project         = var.project_id
  name            = "dgem_epistemic_entropy"
  description     = "Calibrated Epistemic Shannon Entropy (H_max in nats) per DiffusionGemma decision"
  filter          = "resource.type=\"cloud_run_revision\" AND jsonPayload.span_name=\"dgem.gateway.decide\" AND jsonPayload.dgem_max_entropy >= 0"
  value_extractor = "EXTRACT(jsonPayload.dgem_max_entropy)"

  bucket_options {
    linear_buckets {
      num_finite_buckets = 25
      width              = 0.05
      offset             = 0.0
    }
  }

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "1"

    labels {
      key         = "template"
      value_type  = "STRING"
      description = "Decision policy template name"
    }
    labels {
      key         = "surface"
      value_type  = "STRING"
      description = "Access surface"
    }
  }

  label_extractors = {
    template = "EXTRACT(jsonPayload.dgem_template)"
    surface  = "EXTRACT(jsonPayload.dgem_surface)"
  }

  depends_on = [google_project_service.required]
}

resource "google_logging_metric" "dgem_gpu_forward_ms" {
  count           = var.enable_monitoring ? 1 : 0
  project         = var.project_id
  name            = "dgem_gpu_forward_ms"
  description     = "Pure vLLM GPU forward-pass duration (ms) for DiffusionGemma decision policies"
  filter          = "resource.type=\"cloud_run_revision\" AND jsonPayload.span_name=\"dgem.gateway.decide\" AND jsonPayload.dgem_gpu_forward_ms > 0"
  value_extractor = "EXTRACT(jsonPayload.dgem_gpu_forward_ms)"

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 30
      growth_factor      = 1.3
      scale              = 20.0
    }
  }

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "template"
      value_type  = "STRING"
      description = "Decision policy template name"
    }
    labels {
      key         = "surface"
      value_type  = "STRING"
      description = "Access surface"
    }
  }

  label_extractors = {
    template = "EXTRACT(jsonPayload.dgem_template)"
    surface  = "EXTRACT(jsonPayload.dgem_surface)"
  }

  depends_on = [google_project_service.required]
}

resource "google_logging_metric" "dgem_gpu_prefill_ms" {
  count           = var.enable_monitoring ? 1 : 0
  project         = var.project_id
  name            = "dgem_gpu_prefill_ms"
  description     = "vLLM GPU prompt and SigLIP vision prefill duration (ms) for DiffusionGemma decision policies"
  filter          = "resource.type=\"cloud_run_revision\" AND jsonPayload.span_name=\"dgem.gateway.decide\" AND jsonPayload.dgem_gpu_prefill_ms > 0"
  value_extractor = "EXTRACT(jsonPayload.dgem_gpu_prefill_ms)"

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 30
      growth_factor      = 1.3
      scale              = 10.0
    }
  }

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "template"
      value_type  = "STRING"
      description = "Decision policy template name"
    }
    labels {
      key         = "surface"
      value_type  = "STRING"
      description = "Access surface"
    }
  }

  label_extractors = {
    template = "EXTRACT(jsonPayload.dgem_template)"
    surface  = "EXTRACT(jsonPayload.dgem_surface)"
  }

  depends_on = [google_project_service.required]
}

resource "google_logging_metric" "dgem_gpu_denoise_ms" {
  count           = var.enable_monitoring ? 1 : 0
  project         = var.project_id
  name            = "dgem_gpu_denoise_ms"
  description     = "vLLM bidirectional diffusion canvas denoising duration (ms) for DiffusionGemma decision policies"
  filter          = "resource.type=\"cloud_run_revision\" AND jsonPayload.span_name=\"dgem.gateway.decide\" AND jsonPayload.dgem_gpu_denoise_ms > 0"
  value_extractor = "EXTRACT(jsonPayload.dgem_gpu_denoise_ms)"

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 30
      growth_factor      = 1.3
      scale              = 15.0
    }
  }

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "template"
      value_type  = "STRING"
      description = "Decision policy template name"
    }
    labels {
      key         = "surface"
      value_type  = "STRING"
      description = "Access surface"
    }
  }

  label_extractors = {
    template = "EXTRACT(jsonPayload.dgem_template)"
    surface  = "EXTRACT(jsonPayload.dgem_surface)"
  }

  depends_on = [google_project_service.required]
}

resource "google_logging_metric" "dgem_warmup_total_ms" {
  count           = var.enable_monitoring ? 1 : 0
  project         = var.project_id
  name            = "dgem_warmup_total_ms"
  description     = "Total GPU cold-start warmup duration (ms) for right-sizing scale-from-zero"
  filter          = "resource.type=\"cloud_run_revision\" AND jsonPayload.span_name=\"dgem.gpu.warmup_lifecycle\" AND jsonPayload.dgem_warmup_total_ms > 0"
  value_extractor = "EXTRACT(jsonPayload.dgem_warmup_total_ms)"

  bucket_options {
    linear_buckets {
      num_finite_buckets = 30
      width              = 10000.0
      offset             = 10000.0
    }
  }

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"

    labels {
      key         = "trigger"
      value_type  = "STRING"
      description = "Warmup trigger source (web_studio_wake, auto_wake_decide, mcp_agent, api_warmup)"
    }
  }

  label_extractors = {
    trigger = "EXTRACT(jsonPayload.dgem_warmup_trigger)"
  }

  depends_on = [google_project_service.required]
}

resource "google_monitoring_dashboard" "operational" {
  count          = var.enable_monitoring ? 1 : 0
  project        = var.project_id
  dashboard_json = file("${path.module}/../monitoring/dgem-operational-dashboard.json")

  depends_on = [
    google_logging_metric.dgem_decisions_total,
    google_logging_metric.dgem_epistemic_entropy,
    google_logging_metric.dgem_gpu_forward_ms,
    google_logging_metric.dgem_gpu_prefill_ms,
    google_logging_metric.dgem_gpu_denoise_ms,
    google_logging_metric.dgem_warmup_total_ms,
  ]
}
