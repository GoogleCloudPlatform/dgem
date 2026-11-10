# Declarative Terraform Deployment (`deploy/terraform`)

To provision the `dgem` Cloud Run GPU engine, IAP-verified HTTP gateway, least-privilege IAM bindings, Vertex AI Dedicated Endpoint shell, and Cloud Monitoring operational dashboard as infrastructure-as-code, apply this Terraform root module (`hashicorp/google` and `hashicorp/google-beta` `>= 6.12.0, < 7.0.0`).

> [!IMPORTANT]
> Standard Cloud Run GPU (`nvidia-rtx-pro-6000` or `nvidia-l4`), Cloud Storage, Artifact Registry, and Vertex AI Dedicated Endpoint pricing applies while resources are provisioned. Set `cloudrun_gpu_min_instances = 0` for scale-to-zero Cloud Run GPU deployments.

## Resource Scope & Lifecycle Responsibilities

The table below separates resources managed declaratively by Terraform from one-time artifact staging and Vertex AI `v1beta1` model-attachment steps:

| Component | Provisioning Mechanism | Notes |
| :--- | :--- | :--- |
| **Project APIs, Artifact Registry (`dgem`), Weights Bucket (`gs://dgem-weights-<PROJECT>`)** | **Managed in Terraform** ([`storage_and_registry.tf`](./storage_and_registry.tf)) | Enforces `uniform_bucket_level_access = true` and `public_access_prevention = "enforced"`. |
| **Service Accounts & Least-Privilege IAM** (`dgemma-gpu-sa`, `dgemma-gateway-sa`) | **Managed in Terraform** ([`iam.tf`](./iam.tf)) | Scopes `roles/storage.objectViewer` strictly to the weights bucket, `roles/secretmanager.secretAccessor` to named secrets, and `roles/run.invoker` to `allow_member` (blocks `allUsers` and `allAuthenticatedUsers`). |
| **Cloud Run v2 GPU Service (`dgemma`)** | **Managed in Terraform** ([`cloudrun_gpu.tf`](./cloudrun_gpu.tf)) | Gen2 execution environment, `gpu_zonal_redundancy_disabled = true`, Cloud Storage FUSE `/mnt/gcs` volume mount, Direct VPC Egress (`ALL_TRAFFIC`), and `/ready` startup probe (`failure_threshold = 180`). |
| **Cloud Run v2 Gateway (`dgemma-gateway`)** | **Managed in Terraform** ([`gateway.tf`](./gateway.tf)) | Wires `UPSTREAM_DGEMMA_URL`, `DGEM_IAP_AUDIENCE`, `DGEM_VERTEX_URL`, and `DGEM_SERVER_KEY` Secret Manager references. |
| **6 Cloud Logging Metrics + Operational Dashboard** | **Managed in Terraform** ([`monitoring.tf`](./monitoring.tf)) | Creates `dgem_decisions_total`, `dgem_epistemic_entropy`, `dgem_gpu_forward_ms`, `dgem_gpu_prefill_ms`, `dgem_gpu_denoise_ms`, `dgem_warmup_total_ms`, and [`../monitoring/dgem-operational-dashboard.json`](../monitoring/dgem-operational-dashboard.json). |
| **Vertex AI Dedicated Endpoint Shell** (`google_vertex_ai_endpoint`) | **Managed in Terraform** ([`vertex_endpoint.tf`](./vertex_endpoint.tf), `enable_vertex_endpoint = true`) | Provisions the dedicated DNS endpoint (`*.prediction.vertexai.goog`). |
| **Container Image Builds & GCS Weight Staging** | **One-Time CLI Step** (`make cloudrun-build`, [`../../scripts/stage_model_gcs.sh`](../../scripts/stage_model_gcs.sh)) | Run once after creating Artifact Registry and the weights bucket, or pass pre-built `cloudrun_gpu_image` and `gateway_image` URIs. |
| **Vertex `DeployedModel` Attachment (`invokeRoutePrefix = "/*"`)** | **Vertex `v1beta1` REST / Script** ([`../../scripts/deploy_vertex_endpoint.sh`](../../scripts/deploy_vertex_endpoint.sh)) | Custom `invokeRoutePrefix` (`/*`) on `DeployedModel` uses the Vertex `v1beta1` `:deployModel` endpoint (`ENDPOINT_ID=<id> ./scripts/deploy_vertex_endpoint.sh`). |

## Prerequisites

1. Terraform `>= 1.6.0` and Google Cloud Application Default Credentials (`gcloud auth application-default login`).
2. Private Google Access enabled on `cloudrun_vpc_subnet` (`default` by default) when `enable_direct_vpc_egress = true`:

   ```bash
   gcloud compute networks subnets update default \
     --project="PROJECT_ID" \
     --region="us-central1" \
     --enable-private-ip-google-access
   ```

   Replace the following:
   - `PROJECT_ID`: the ID of your Google Cloud project.

3. Optional Secret Manager secrets created ahead of time if setting `hf_token_secret_id` or `server_key_secret_id`:

   ```bash
   openssl rand -base64 32 | tr -d '\n' | \
     gcloud secrets create dgem-server-key \
       --project="PROJECT_ID" \
       --data-file=-
   ```

## Deploy in Two Stages (Fresh Project)

On a brand-new project where container images and weights have not been staged yet, apply storage and registry targets first, push images and weights, and then run a full `terraform apply`:

1. Copy [`terraform.tfvars.example`](./terraform.tfvars.example) to `terraform.tfvars` and set `project_id` and `allow_member`.
2. Provision APIs, Artifact Registry, and the weights bucket:

   ```bash
   terraform -chdir=deploy/terraform init
   terraform -chdir=deploy/terraform apply \
     -target=google_project_service.required \
     -target=google_artifact_registry_repository.dgem \
     -target=google_storage_bucket.weights
   ```

3. Stage model weights into Cloud Storage and build the serving and gateway images:

   ```bash
   ./scripts/stage_model_gcs.sh "dgem-weights-PROJECT_ID" "us-central1"
   GCP_PROJECT="PROJECT_ID" make cloudrun-build
   ```

4. Apply the complete stack (Cloud Run GPU, Gateway, IAM, and Monitoring):

   ```bash
   terraform -chdir=deploy/terraform apply
   ```

## Offline Configuration Verification

Run the hermetic HCL structural, reference, dashboard-metric, and secret-hygiene test suite without requiring cloud credentials:

```bash
python3 -m unittest -v deploy/terraform/test_terraform_config.py
```
