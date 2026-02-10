# Terraform — GCP Infrastructure

Infrastructure as Code for the AI Product Photo Detector MLOps pipeline on Google Cloud Platform.

## Architecture

| Resource | Purpose |
|---|---|
| **GCS Bucket** | DVC remote storage for datasets and model artifacts |
| **Artifact Registry** | Docker image repository for the FastAPI container |
| **Cloud Run** | Serverless deployment of the inference API |
| **Service Account** | Least-privilege identity for Cloud Run and CI/CD |

## Prerequisites

1. **Google Cloud SDK** — [Install gcloud](https://cloud.google.com/sdk/docs/install)
2. **Terraform** ≥ 1.5 — [Install Terraform](https://developer.hashicorp.com/terraform/install)
3. A GCP project with billing enabled
4. Authenticated `gcloud` session:

```bash
gcloud auth login
gcloud auth application-default login
```

## Quick Start

### 1. Configure variables

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and set your `project_id` at minimum.

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Review the execution plan

```bash
terraform plan
```

### 4. Apply the infrastructure

```bash
terraform apply
```

Type `yes` when prompted. Terraform will create all resources and display the outputs.

### 5. Verify outputs

```bash
terraform output
```

Expected outputs:
- `cloud_run_url` — API endpoint
- `gcs_bucket_name` — DVC remote bucket
- `artifact_registry_url` — Docker push target
- `service_account_email` — SA for CI/CD configuration

## Post-Deployment Setup

### Configure DVC remote

```bash
dvc remote modify gcs url gs://$(terraform output -raw gcs_bucket_name)
```

### Push a Docker image

```bash
REGISTRY=$(terraform output -raw artifact_registry_url)

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker $(echo $REGISTRY | cut -d/ -f1)

# Build and push
docker build -t $REGISTRY/ai-product-detector:latest -f ../docker/Dockerfile ..
docker push $REGISTRY/ai-product-detector:latest
```

### Update Cloud Run with a new image

```bash
gcloud run services update ai-product-detector \
  --image $REGISTRY/ai-product-detector:latest \
  --region europe-west1
```

## Environment Promotion

Use the `environment` variable to manage separate environments:

```bash
# Dev (default)
terraform workspace new dev
terraform apply -var="environment=dev"

# Production
terraform workspace new prod
terraform apply -var="environment=prod" \
  -var="cloud_run_min_instances=1" \
  -var="cloud_run_max_instances=10"
```

## Destroy

```bash
terraform destroy
```

> ⚠️ The GCS bucket has `force_destroy = false` by default. Empty it first or set `force_destroy = true` in `main.tf` before destroying.

## File Structure

```
terraform/
├── main.tf                  # Provider, resources (GCS, AR, Cloud Run, IAM)
├── variables.tf             # Input variables with validation
├── outputs.tf               # Useful outputs for downstream tools
├── terraform.tfvars.example # Template for variable values
└── README.md                # This file
```
