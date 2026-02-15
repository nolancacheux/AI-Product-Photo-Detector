# GCP Infrastructure and Terraform

All cloud infrastructure is defined as code using Terraform in the `terraform/` directory. This document covers every provisioned resource, setup instructions, cost considerations, and teardown procedures.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Structure](#module-structure)
3. [Environment Comparison](#environment-comparison)
4. [GCP Resources](#gcp-resources)
5. [Terraform Setup](#terraform-setup)
6. [Configuration Reference](#configuration-reference)
7. [Service Account Permissions](#service-account-permissions)
8. [Cost Estimation](#cost-estimation)
9. [Remote State](#remote-state)
10. [Teardown](#teardown)

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

## Module Structure

The Terraform configuration follows a modular architecture with per-environment configurations:

```
terraform/
├── environments/          # Per-environment configurations
│   ├── dev/               # Development (scale-to-zero, 512Mi, 10€ budget)
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars
│   └── prod/              # Production (min 1 instance, 1Gi, 50€ budget)
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       └── terraform.tfvars
├── modules/               # Reusable infrastructure modules
│   ├── cloud-run/
│   ├── storage/
│   ├── registry/
│   ├── monitoring/
│   └── iam/
├── backend.tf
├── versions.tf
└── README.md
```

### Modules Overview

| Module | Purpose |
|--------|---------|
| `cloud-run` | Cloud Run v2 service with configurable scaling, resources, and health probes |
| `storage` | GCS bucket with versioning, lifecycle rules, and enforced public access prevention |
| `registry` | Artifact Registry Docker repository with automatic cleanup policies |
| `monitoring` | Uptime checks on `/health`, alert policies for downtime and error rate |
| `iam` | Service account with least-privilege roles for Cloud Run |

---

## Environment Comparison

| Setting               | Dev                    | Prod                        |
|-----------------------|------------------------|-----------------------------|
| Min instances         | 0 (scale-to-zero)      | 1 (always-on)               |
| Max instances         | 2                      | 10                          |
| Memory                | 512Mi                  | 1Gi                         |
| Budget                | 10€/month              | 50€/month                   |
| Monitoring            | Optional (off)         | Always enabled              |
| Bucket force_destroy  | true                   | false                       |
| Image retention       | 5 recent / 3d untagged | 20 recent / 14d untagged    |
| Custom domain         | N/A                    | Supported                   |

---

## GCP Resources

### Enabled APIs

Terraform enables the following GCP APIs automatically:

| API | Purpose |
|-----|---------|
| `run.googleapis.com` | Cloud Run service deployment |
| `artifactregistry.googleapis.com` | Docker image storage |
| `storage.googleapis.com` | GCS bucket operations |
| `iam.googleapis.com` | Service account and role management |
| `cloudresourcemanager.googleapis.com` | Project-level resource management |
| `billingbudgets.googleapis.com` | Budget alerts |
| `monitoring.googleapis.com` | Uptime checks and alerting |

### Google Cloud Storage Bucket

| Property | Value |
|----------|-------|
| Module | `modules/storage` |
| Name | `<PROJECT_ID>-mlops-data` |
| Location | Same as `var.region` (default: `europe-west1`) |
| Access | Uniform bucket-level, public access prevented |
| Versioning | Enabled |
| Lifecycle | Configurable version retention and archive cleanup |

**Purpose:** Stores DVC-tracked training data, model checkpoints, and MLflow artifacts.

### Artifact Registry Repository

| Property | Value |
|----------|-------|
| Module | `modules/registry` |
| Name | `ai-product-detector` |
| Format | Docker |
| Cleanup | Configurable retention for recent and untagged images |

**Purpose:** Stores Docker images for both the inference API and the training container.

### Cloud Run Service

| Property | Value |
|----------|-------|
| Module | `modules/cloud-run` |
| Name | `ai-product-detector` |
| Container port | 8080 |
| CPU | Configurable (default: `1000m` = 1 vCPU) |
| Memory | Configurable (default: `512Mi` in dev, `1Gi` in prod) |
| Min instances | Configurable (default: `0` in dev, `1` in prod) |
| Max instances | Configurable (default: `2` in dev, `10` in prod) |
| Service account | Dedicated least-privilege SA |
| Public access | Configurable (default: unauthenticated) |

**Health Probes:**

| Probe | Path | Config |
|-------|------|--------|
| Startup | `/startup` | Initial delay, failure threshold configurable |
| Liveness | `/healthz` | Periodic check that process is alive |
| Readiness | `/readyz` | Checks model is loaded and ready |

**Environment variables injected:**

- `ENVIRONMENT` - deployment environment (dev/staging/prod)
- `GCS_BUCKET` - bucket name for model/data access

### IAM Service Account

| Property | Value |
|----------|-------|
| Module | `modules/iam` |
| Account ID | `ai-product-detector-sa` |

| IAM Role | Purpose |
|----------|---------|
| `roles/storage.objectAdmin` | Read/write GCS objects |
| `roles/artifactregistry.reader` | Pull Docker images |
| `roles/logging.logWriter` | Write application logs |
| `roles/monitoring.metricWriter` | Write custom metrics |

### Monitoring (Production)

| Property | Value |
|----------|-------|
| Module | `modules/monitoring` |
| Uptime check | HTTP GET `/health` every 60s |
| Downtime alert | Fires after configurable duration |
| Error rate alert | Fires when 5xx rate exceeds threshold |
| Notifications | Email to configured recipients |

### Billing Budget Alert

| Property | Value |
|----------|-------|
| Budget amount | 10€ (dev) / 50€ (prod) |
| Alert thresholds | 50%, 80%, 100% of budget |

---

## Terraform Setup

### Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- A GCP project with billing enabled
- A service account key or `gcloud auth application-default login`

### Step-by-Step

#### 1. Authenticate

```bash
# Option A: Application Default Credentials (recommended for local use)
gcloud auth application-default login

# Option B: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa-key.json"
```

#### 2. Choose your environment

```bash
# Development
cd terraform/environments/dev

# Or production
cd terraform/environments/prod
```

#### 3. Configure variables

```bash
vim terraform.tfvars
```

At minimum, set `project_id`:

```hcl
project_id = "<YOUR-PROJECT-ID>"
```

#### 4. Initialize Terraform

```bash
terraform init
```

#### 5. Preview changes

```bash
terraform plan
```

#### 6. Apply

```bash
terraform apply
```

Type `yes` when prompted. Terraform will provision all resources and print outputs including the Cloud Run URL, bucket name, and registry URL.

#### 7. Verify outputs

```bash
terraform output
```

---

## Configuration Reference

### Module: cloud-run

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `cpu` | string | `1000m` | CPU allocation (1000m = 1 vCPU) |
| `memory` | string | `512Mi` | Memory allocation |
| `min_instances` | number | `0` | Min instances (0 = scale-to-zero) |
| `max_instances` | number | `2` | Maximum instances |
| `allow_unauthenticated` | bool | `true` | Public access |

### Module: storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `force_destroy` | bool | `false` | Allow bucket deletion with objects |
| `versioning_max_versions` | number | `5` | Versions to keep |
| `archive_retention_days` | number | `90` | Archive retention |

### Module: registry

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `keep_count` | number | `10` | Recent images to keep |
| `untagged_max_age_seconds` | number | `604800` | Max untagged image age (7d) |

### Module: monitoring

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `enable_monitoring` | bool | `true` | Enable/disable all monitoring |
| `health_check_path` | string | `/health` | HTTP path to check |
| `alert_downtime_duration` | string | `60s` | Downtime before alert |
| `error_rate_threshold` | number | `5` | 5xx error rate % threshold |

---

## Service Account Permissions

### Terraform Execution

The identity running `terraform apply` needs:

- `roles/editor` or a combination of:
  - `roles/run.admin`
  - `roles/artifactregistry.admin`
  - `roles/storage.admin`
  - `roles/iam.serviceAccountAdmin`
  - `roles/iam.projectIamAdmin`
  - `roles/serviceusage.serviceUsageAdmin`
  - `roles/billing.viewer` (if using budget alerts)
  - `roles/monitoring.admin` (if using monitoring module)

### CI/CD Service Account (GitHub Actions)

The service account key stored in `GCP_SA_KEY` needs:

| Role | Purpose |
|------|---------|
| `roles/run.admin` | Deploy Cloud Run services |
| `roles/artifactregistry.writer` | Push Docker images |
| `roles/storage.objectAdmin` | Read/write GCS data and models |
| `roles/aiplatform.user` | Submit Vertex AI training jobs |
| `roles/iam.serviceAccountUser` | Act as the Cloud Run service account |

### Runtime Service Account (Cloud Run)

Provisioned by Terraform with minimal permissions:

| Role | Purpose |
|------|---------|
| `roles/storage.objectAdmin` | Read model checkpoints and data |
| `roles/artifactregistry.reader` | Pull container images |
| `roles/logging.logWriter` | Application logging |
| `roles/monitoring.metricWriter` | Metrics export |

---

## Cost Estimation

These estimates assume a small-scale project with minimal traffic.

### Cloud Run

| Component | Cost | Notes |
|-----------|------|-------|
| CPU (idle) | Free | Scale-to-zero with `min_instances = 0` |
| CPU (active) | ~$0.00002400/vCPU-second | Billed only when handling requests |
| Memory (active) | ~$0.00000250/GiB-second | |
| Free tier | 2M requests/month, 360K vCPU-seconds | Generous free tier covers most light usage |

**Estimated monthly cost (low traffic):** $0 – $2

### Google Cloud Storage

| Component | Cost | Notes |
|-----------|------|-------|
| Storage (Standard) | ~$0.020/GB/month | Training data + model checkpoints |
| Operations | ~$0.005 per 1K Class A ops | Writes |
| Egress | Free within same region | Cross-region egress charged |

**Estimated monthly cost (10 GB data):** ~$0.20

### Artifact Registry

| Component | Cost | Notes |
|-----------|------|-------|
| Storage | ~$0.10/GB/month | Docker images (cleanup policies help) |

**Estimated monthly cost (5 images):** ~$0.50

### Vertex AI Training

| Component | Cost | Notes |
|-----------|------|-------|
| `n1-standard-4` | ~$0.19/hour | 4 vCPUs, 15 GB RAM |
| NVIDIA Tesla T4 | ~$0.35/hour | 1 GPU |
| **Total per hour** | **~$0.54/hour** | |

**Estimated cost per training run (1 hour):** ~$0.54

### Budget Alert

The Terraform configuration includes a budget alert (default: 10€/month for dev, 50€/month for prod) with notifications at 50%, 80%, and 100% thresholds.

### Total Estimated Monthly Cost

| Scenario | Estimate |
|----------|----------|
| Development (occasional training, low traffic) | $1 – $5 |
| Active development (weekly training, moderate traffic) | $5 – $15 |

---

## Remote State

By default, Terraform stores state locally. For team collaboration, enable GCS remote state.

### Setup (one-time)

```bash
PROJECT_ID="<YOUR-PROJECT-ID>"

# Create state bucket
gsutil mb -l europe-west1 gs://${PROJECT_ID}-tfstate
gsutil versioning set on gs://${PROJECT_ID}-tfstate
```

### Enable

Uncomment the `backend "gcs"` block in your environment's `main.tf`:

```hcl
terraform {
  backend "gcs" {
    bucket = "<YOUR-PROJECT-ID>-tfstate"
    prefix = "terraform/state/dev"  # or "terraform/state/prod"
  }
}
```

Then migrate:

```bash
terraform init -migrate-state
```

---

## Teardown

### Destroy all Terraform-managed resources

```bash
cd terraform/environments/dev  # or prod
terraform destroy
```

Type `yes` when prompted. This removes:
- Cloud Run service
- Artifact Registry repository (and all images)
- IAM service account and role bindings
- Monitoring resources (uptime checks, alerts)
- Billing budget alert

**Note:** The GCS bucket has `force_destroy = false` by default in production, meaning Terraform will refuse to delete it if it contains objects. To force deletion:

```bash
# Empty the bucket first
gsutil -m rm -r gs://<YOUR-PROJECT-ID>-mlops-data/**

# Then destroy
terraform destroy
```

### Manual cleanup

If Terraform state becomes inconsistent, remove resources manually:

```bash
# Delete Cloud Run service
gcloud run services delete <SERVICE-NAME> --region=<REGION>

# Delete Artifact Registry repository
gcloud artifacts repositories delete <REPO-NAME> --location=<REGION>

# Delete GCS bucket
gsutil -m rm -r gs://<YOUR-PROJECT-ID>-mlops-data
gsutil rb gs://<YOUR-PROJECT-ID>-mlops-data

# Delete service account
gcloud iam service-accounts delete \
  ai-product-detector-sa@<YOUR-PROJECT-ID>.iam.gserviceaccount.com
```

### Disable APIs (optional)

```bash
gcloud services disable run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  monitoring.googleapis.com
```

This is usually unnecessary and may affect other resources in the project.

---

## Common Operations

### Deploy a new image

```bash
IMAGE="<REGION>-docker.pkg.dev/<YOUR-PROJECT-ID>/ai-product-detector/api:v1.0.0"
docker build -f docker/Dockerfile -t $IMAGE .
docker push $IMAGE

# Deploy via Terraform
cd terraform/environments/prod
terraform apply -var="cloud_run_container_image=$IMAGE"
```

### Format all Terraform files

```bash
terraform fmt -recursive terraform/
```

### Validate configuration

```bash
cd terraform/environments/dev
terraform validate
```

### Import existing resource

```bash
terraform import module.cloud_run.google_cloud_run_v2_service.api \
  projects/<YOUR-PROJECT-ID>/locations/<REGION>/services/<SERVICE-NAME>
```
