# GCP Infrastructure and Terraform

All cloud infrastructure is defined as code using Terraform in the `terraform/`
directory. This document covers every provisioned resource, setup instructions,
cost considerations, and teardown procedures.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [GCP Resources](#gcp-resources)
3. [Terraform Setup](#terraform-setup)
4. [Configuration Reference](#configuration-reference)
5. [Service Account Permissions](#service-account-permissions)
6. [Cost Estimation](#cost-estimation)
7. [Remote State (Optional)](#remote-state-optional)
8. [Teardown](#teardown)

---

## Architecture Overview

```
                           +---------------------------+
                           |       GitHub Actions      |
                           |  (CI / CD / Training)     |
                           +-----+----------+----------+
                                 |          |
                    Push image   |          |  Submit job
                                 v          v
+-------------------+   +----------------+   +------------------+
| Artifact Registry |   |   Cloud Run    |   |   Vertex AI      |
| (Docker images)   |   |  (Inference)   |   | (GPU Training)   |
+-------------------+   +-------+--------+   +--------+---------+
                                |                     |
                          Reads model           Reads data /
                          at build time         Writes model
                                |                     |
                                v                     v
                        +-----------------------------+
                        |     Google Cloud Storage     |
                        |  (DVC data + model storage)  |
                        +-----------------------------+
                                       |
                                       | Managed by
                                       v
                        +-----------------------------+
                        |     IAM Service Account     |
                        | (least-privilege identity)  |
                        +-----------------------------+
                                       |
                                       | Alerts
                                       v
                        +-----------------------------+
                        |     Billing Budget Alert    |
                        +-----------------------------+
```

---

## GCP Resources

### Enabled APIs

Terraform enables the following GCP APIs automatically:

| API | Purpose |
|---|---|
| `run.googleapis.com` | Cloud Run service deployment |
| `artifactregistry.googleapis.com` | Docker image storage |
| `storage.googleapis.com` | GCS bucket operations |
| `iam.googleapis.com` | Service account and role management |
| `cloudresourcemanager.googleapis.com` | Project-level resource management |
| `billingbudgets.googleapis.com` | Budget alerts |

**Source:** `terraform/main.tf` -- `google_project_service.required_apis`

### Google Cloud Storage Bucket

| Property | Value |
|---|---|
| Resource | `google_storage_bucket.mlops_data` |
| Name | `<project_id>-mlops-data` |
| Location | Same as `var.region` (default: `europe-west1`) |
| Access | Uniform bucket-level, public access prevented |
| Versioning | Enabled |
| Lifecycle | Delete after 5 newer versions; delete archived objects after 90 days |

**Purpose:** Stores DVC-tracked training data, model checkpoints, and MLflow
artifacts.

### Artifact Registry Repository

| Property | Value |
|---|---|
| Resource | `google_artifact_registry_repository.docker_repo` |
| Name | `ai-product-detector` |
| Format | Docker |
| Cleanup | Keep 10 most recent versions; delete untagged images after 7 days |

**Purpose:** Stores Docker images for both the inference API and the training
container.

### Cloud Run Service

| Property | Value |
|---|---|
| Resource | `google_cloud_run_v2_service.api` |
| Name | `ai-product-detector` |
| Container port | 8000 |
| CPU | Configurable (default: `1000m` = 1 vCPU) |
| Memory | Configurable (default: `512Mi`) |
| Min instances | Configurable (default: `0` -- scale to zero) |
| Max instances | Configurable (default: `2`) |
| Service account | Dedicated least-privilege SA |
| Public access | Unauthenticated (`allUsers` as `roles/run.invoker`) |

**Probes:**

| Probe | Path | Config |
|---|---|---|
| Startup | `/health` | 5s initial delay, 10s period, 3 failure threshold |
| Liveness | `/health` | 30s period |

**Environment variables injected:**

- `ENVIRONMENT` -- deployment environment (dev/staging/prod)
- `GCS_BUCKET` -- bucket name for model/data access

### IAM Service Account

| Property | Value |
|---|---|
| Resource | `google_service_account.app_sa` |
| Account ID | `ai-product-detector-sa` |
| Roles | See table below |

| IAM Role | Purpose |
|---|---|
| `roles/storage.objectAdmin` | Read/write GCS objects |
| `roles/artifactregistry.reader` | Pull Docker images |
| `roles/logging.logWriter` | Write application logs |
| `roles/monitoring.metricWriter` | Write custom metrics |

### Billing Budget Alert

| Property | Value |
|---|---|
| Resource | `google_billing_budget.monthly_budget` |
| Currency | EUR |
| Default amount | 10 EUR |
| Alert thresholds | 50%, 80%, 100% of budget |
| Condition | Requires `billing_account` variable to be set |

---

## Terraform Setup

### Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- A GCP project with billing enabled
- A service account key or `gcloud auth application-default login`

### Step-by-step

#### 1. Authenticate

```bash
# Option A: Application Default Credentials (recommended for local use)
gcloud auth application-default login

# Option B: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"
```

#### 2. Create the variables file

```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your project ID and desired configuration. At
minimum, set `project_id`:

```hcl
project_id = "my-gcp-project-id"
```

#### 3. Initialize Terraform

```bash
terraform init
```

This downloads the Google provider plugin and initializes the working directory.

#### 4. Preview changes

```bash
terraform plan
```

Review the output. Terraform will show every resource it intends to create.

#### 5. Apply

```bash
terraform apply
```

Type `yes` when prompted. Terraform will provision all resources and print
outputs including the Cloud Run URL, bucket name, and registry URL.

#### 6. Verify outputs

```bash
terraform output
```

Expected outputs:

| Output | Description |
|---|---|
| `project_id` | GCP project ID |
| `region` | GCP region |
| `cloud_run_url` | Deployed Cloud Run service URL |
| `cloud_run_service_name` | Cloud Run service name |
| `gcs_bucket_name` | GCS bucket name |
| `gcs_bucket_url` | GCS bucket URL (`gs://...`) |
| `artifact_registry_url` | Artifact Registry repository URL |
| `service_account_email` | Cloud Run service account email |
| `docker_push_command` | Example Docker push command |

---

## Configuration Reference

All variables are defined in `terraform/variables.tf`.

| Variable | Type | Default | Description |
|---|---|---|---|
| `project_id` | string | (required) | GCP project ID |
| `region` | string | `europe-west1` | GCP region |
| `app_name` | string | `ai-product-detector` | Application name for resource naming |
| `environment` | string | `dev` | Environment (`dev`, `staging`, `prod`) |
| `cloud_run_container_image` | string | `""` | Custom container image (empty = use Artifact Registry default) |
| `cloud_run_cpu` | string | `1000m` | CPU allocation (1000m = 1 vCPU) |
| `cloud_run_memory` | string | `512Mi` | Memory allocation |
| `cloud_run_max_instances` | number | `2` | Maximum Cloud Run instances (1--10) |
| `cloud_run_min_instances` | number | `0` | Minimum instances (0 = scale to zero) |
| `billing_account` | string | `""` | Billing account ID (empty = skip budget) |
| `budget_amount` | number | `10` | Monthly budget alert threshold (EUR) |

---

## Service Account Permissions

### Terraform Execution

The identity running `terraform apply` (your user account or a CI service
account) needs:

- `roles/editor` or a combination of:
  - `roles/run.admin`
  - `roles/artifactregistry.admin`
  - `roles/storage.admin`
  - `roles/iam.serviceAccountAdmin`
  - `roles/iam.projectIamAdmin`
  - `roles/serviceusage.serviceUsageAdmin`
  - `roles/billing.viewer` (if using budget alerts)

### CI/CD Service Account (GitHub Actions)

The service account key stored in `GCP_SA_KEY` needs:

| Role | Purpose |
|---|---|
| `roles/run.admin` | Deploy Cloud Run services |
| `roles/artifactregistry.writer` | Push Docker images |
| `roles/storage.objectAdmin` | Read/write GCS data and models |
| `roles/aiplatform.user` | Submit Vertex AI training jobs |
| `roles/iam.serviceAccountUser` | Act as the Cloud Run service account |

### Runtime Service Account (Cloud Run)

Provisioned by Terraform with minimal permissions:

| Role | Purpose |
|---|---|
| `roles/storage.objectAdmin` | Read model checkpoints and data |
| `roles/artifactregistry.reader` | Pull container images |
| `roles/logging.logWriter` | Application logging |
| `roles/monitoring.metricWriter` | Metrics export |

---

## Cost Estimation

These estimates assume a student or small-scale project with minimal traffic.

### Cloud Run

| Component | Cost | Notes |
|---|---|---|
| CPU (idle) | Free | Scale-to-zero with `min_instances = 0` |
| CPU (active) | ~$0.00002400/vCPU-second | Billed only when handling requests |
| Memory (active) | ~$0.00000250/GiB-second | |
| Free tier | 2M requests/month, 360K vCPU-seconds, 180K GiB-seconds | Generous free tier covers most student usage |

**Estimated monthly cost (low traffic):** $0 -- $2

### Google Cloud Storage

| Component | Cost | Notes |
|---|---|---|
| Storage (Standard) | ~$0.020/GB/month | Training data + model checkpoints |
| Operations | ~$0.005 per 1K Class A ops | Writes |
| Egress | Free within same region | Cross-region egress charged |

**Estimated monthly cost (10 GB data):** ~$0.20

### Artifact Registry

| Component | Cost | Notes |
|---|---|---|
| Storage | ~$0.10/GB/month | Docker images (cleanup policies help) |

**Estimated monthly cost (5 images):** ~$0.50

### Vertex AI Training

| Component | Cost | Notes |
|---|---|---|
| `n1-standard-4` | ~$0.19/hour | 4 vCPUs, 15 GB RAM |
| NVIDIA Tesla T4 | ~$0.35/hour | 1 GPU |
| **Total per hour** | **~$0.54/hour** | |

**Estimated cost per training run (1 hour):** ~$0.54

### Budget Alert

The Terraform configuration includes a budget alert (default: 10 EUR/month) with
notifications at 50%, 80%, and 100% thresholds.

### Total Estimated Monthly Cost

| Scenario | Estimate |
|---|---|
| Development (occasional training, low traffic) | $1 -- $5 |
| Active development (weekly training, moderate traffic) | $5 -- $15 |

---

## Remote State (Optional)

By default, Terraform stores state locally in `terraform/terraform.tfstate`.
For team environments, enable remote state in GCS:

#### 1. Create the state bucket

```bash
gsutil mb -l europe-west1 gs://<PROJECT_ID>-tfstate
gsutil versioning set on gs://<PROJECT_ID>-tfstate
```

#### 2. Uncomment the backend block

In `terraform/main.tf`, uncomment the `backend "gcs"` block and set your bucket
name:

```hcl
terraform {
  backend "gcs" {
    bucket = "<PROJECT_ID>-tfstate"
    prefix = "terraform/state"
  }
}
```

#### 3. Re-initialize

```bash
terraform init -migrate-state
```

---

## Teardown

### Destroy all Terraform-managed resources

```bash
cd terraform/
terraform destroy
```

Type `yes` when prompted. This removes:
- Cloud Run service
- Artifact Registry repository (and all images)
- IAM service account and role bindings
- Billing budget alert

**Note:** The GCS bucket has `force_destroy = false` by default, meaning
Terraform will refuse to delete it if it contains objects. To force deletion:

```bash
# Empty the bucket first
gsutil -m rm -r gs://<PROJECT_ID>-mlops-data/**

# Then destroy
terraform destroy
```

Alternatively, set `force_destroy = true` in `main.tf` before destroying.

### Manual cleanup

If Terraform state becomes inconsistent, remove resources manually:

```bash
# Delete Cloud Run service
gcloud run services delete ai-product-detector --region=europe-west1

# Delete Artifact Registry repository
gcloud artifacts repositories delete ai-product-detector \
  --location=europe-west1

# Delete GCS bucket
gsutil -m rm -r gs://<PROJECT_ID>-mlops-data
gsutil rb gs://<PROJECT_ID>-mlops-data

# Delete service account
gcloud iam service-accounts delete \
  ai-product-detector-sa@<PROJECT_ID>.iam.gserviceaccount.com
```

### Disable APIs (optional)

```bash
gcloud services disable run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

This is usually unnecessary and may affect other resources in the project.
