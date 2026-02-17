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
│   ├── dev/               # Development (scale-to-zero, 512Mi, 10 EUR budget)
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars
│   └── prod/              # Production (scale-to-zero, 1Gi, 50 EUR budget)
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
├── backend.tf             # Remote state documentation
├── versions.tf            # Required providers (google ~> 5.0)
├── .gitignore
└── README.md
```

### Modules Overview

| Module | Purpose |
|--------|---------|
| `cloud-run` | Cloud Run v2 service with configurable scaling, resources, and TCP startup probe |
| `storage` | GCS bucket with optional versioning, lifecycle rules for temp files and noncurrent versions |
| `registry` | Artifact Registry Docker repository with automatic cleanup policies |
| `monitoring` | Uptime checks on `/health`, alert policies for downtime and 5xx error rate |
| `iam` | Service account with least-privilege roles for Cloud Run |

---

## Environment Comparison

| Setting               | Dev                    | Prod                        |
|-----------------------|------------------------|-----------------------------|
| Min instances         | 0 (scale-to-zero)      | 0 (scale-to-zero)           |
| Max instances         | 2                      | 3                           |
| CPU                   | 1000m (1 vCPU)         | 1 (1 vCPU)                  |
| Memory                | 512Mi                  | 1Gi                         |
| Budget                | 10 EUR/month           | 50 EUR/month                |
| Monitoring            | Optional (off by default) | Always enabled            |
| Bucket force_destroy  | true                   | false                       |
| Bucket versioning     | Default (false)        | false (explicit)            |
| Image retention       | 5 recent / 3d untagged | 20 recent / 14d untagged    |
| Custom domain         | N/A                    | Supported (optional)        |
| Service name          | `ai-product-detector-dev` (default) | `ai-product-detector` (overridden) |
| Remote state          | Local (GCS commented out) | GCS (`ai-product-detector-487013-tfstate`) |
| SA account ID         | `ai-product-detector-dev` (default) | `mlops-deployer` (overridden) |
| Service account used  | IAM module SA          | Default Compute SA (hardcoded) |
| Env vars              | (via extra_env_vars)   | `REQUIRE_AUTH=false`, `ENVIRONMENT=production` |

---

## GCP Resources

### Enabled APIs

Terraform enables the following GCP APIs automatically (via `google_project_service`):

| API | Purpose |
|-----|---------|
| `run.googleapis.com` | Cloud Run service deployment |
| `artifactregistry.googleapis.com` | Docker image storage |
| `storage.googleapis.com` | GCS bucket operations |
| `iam.googleapis.com` | Service account and role management |
| `cloudresourcemanager.googleapis.com` | Project-level resource management |
| `billingbudgets.googleapis.com` | Budget alerts |
| `monitoring.googleapis.com` | Uptime checks and alerting |

All APIs are set with `disable_on_destroy = false` to avoid disrupting other project resources during teardown.

### Google Cloud Storage Bucket

| Property | Dev | Prod |
|----------|-----|------|
| Module | `modules/storage` | `modules/storage` |
| Name | `<project_id>-ai-product-detector-dev` (default) | `<project_id>-mlops-data` (overridden) |
| Location | `var.region` (default: `europe-west1`) | `var.region` (default: `europe-west1`) |
| Access | Uniform bucket-level | Uniform bucket-level |
| Public access prevention | `inherited` (default) | `inherited` (explicit) |
| Versioning | false (default) | false (explicit) |
| force_destroy | true | false |
| Temp file retention | 90 days (default) | 90 days |
| Temp file prefixes | `tmp/`, `temp/`, `cache/` (default) | `tmp/`, `temp/`, `cache/` |
| Noncurrent version retention | 30 days (default) | 30 days |

**Purpose:** Stores DVC-tracked training data, model checkpoints, and MLflow artifacts.

### Artifact Registry Repository

| Property | Value |
|----------|-------|
| Module | `modules/registry` |
| Repository ID | `ai-product-detector` (from `var.app_name`) |
| Format | Docker |
| Cleanup dry run | Disabled (policies are enforced) |
| Keep recent (dev) | 5 images |
| Keep recent (prod) | 20 images |
| Untagged max age (dev) | 259,200s (3 days) |
| Untagged max age (prod) | 1,209,600s (14 days) |

**Purpose:** Stores Docker images for the inference API and training containers.

### Cloud Run Service

| Property | Dev | Prod |
|----------|-----|------|
| Module | `modules/cloud-run` | `modules/cloud-run` |
| Service name | `ai-product-detector-dev` (default) | `ai-product-detector` (overridden) |
| Container port | 8080 | 8080 |
| CPU | `1000m` (1 vCPU) | `1` (1 vCPU) |
| Memory | `512Mi` | `1Gi` |
| Min instances | 0 (scale-to-zero) | 0 (scale-to-zero) |
| Max instances | 2 | 3 |
| Service account | IAM module SA | `714127049161-compute@developer.gserviceaccount.com` |
| Public access | Unauthenticated (allUsers) | Unauthenticated (allUsers) |
| Startup probe timeout | 240s (default) | 240s |

**Container image resolution:** If `container_image` is empty, defaults to `<region>-docker.pkg.dev/<project_id>/<registry_repo_id>/<image_name>:latest`.

**Startup probe:** TCP socket check on the container port. No HTTP liveness or readiness probes are configured; only the startup probe ensures the container is listening before receiving traffic.

**Environment variables (prod):**

| Variable | Value |
|----------|-------|
| `REQUIRE_AUTH` | `false` |
| `ENVIRONMENT` | `production` |

Environment variables are injected via the `extra_env_vars` map. Dev does not define any extra env vars by default.

### IAM Service Account

| Property | Dev | Prod |
|----------|-----|------|
| Module | `modules/iam` | `modules/iam` |
| Account ID | `ai-product-detector-dev` (default) | `mlops-deployer` (overridden) |
| Display name | Auto-generated from app_name and environment | Auto-generated |

**Base IAM roles (applied to both environments):**

| IAM Role | Purpose |
|----------|---------|
| `roles/storage.objectAdmin` | Read/write GCS objects |
| `roles/artifactregistry.reader` | Pull Docker images |
| `roles/logging.logWriter` | Write application logs |
| `roles/monitoring.metricWriter` | Write custom metrics |

Additional roles can be added via the `additional_roles` variable (used in prod via `additional_iam_roles`).

**Note:** In production, the Cloud Run service currently uses the default Compute Engine service account (`714127049161-compute@developer.gserviceaccount.com`) rather than the IAM module's service account. The IAM module still provisions the `mlops-deployer` SA with the roles listed above for use by CI/CD pipelines.

### Monitoring (Production)

| Property | Value |
|----------|-------|
| Module | `modules/monitoring` |
| Uptime check | HTTPS GET on `/health`, port 443, SSL validated, every 60s |
| Downtime alert | Fires after 60s of consecutive failures |
| Error rate alert | Fires when 5xx request count exceeds threshold (default: 5) |
| Notifications | Email to configured recipients (via `notification_email` variable) |
| Auto-close | Alerts auto-close after 1800s (30 min) |
| Alert documentation | Includes direct link to Cloud Run logs in GCP Console |

In dev, monitoring is disabled by default (`enable_monitoring = false`). It can be enabled by setting `enable_monitoring = true` in `terraform.tfvars`.

### Billing Budget Alert

| Property | Dev | Prod |
|----------|-----|------|
| Budget amount | 10 EUR | 50 EUR |
| Currency | EUR | EUR |
| Alert thresholds | 50%, 80%, 100% of budget | 50%, 80%, 100% of budget |
| Spend basis | CURRENT_SPEND | CURRENT_SPEND |
| Condition | Requires `billing_account` variable to be set | Requires `billing_account` variable to be set |

Budget alerts are only created if the `billing_account` variable is provided. Both dev and prod `terraform.tfvars` templates have it commented out by default.

### Custom Domain Mapping (Prod Only)

Production supports an optional Cloud Run domain mapping via the `custom_domain` variable. When set, Terraform creates a `google_cloud_run_domain_mapping` resource that routes the custom domain to the Cloud Run service. DNS verification is required. This is not available in the dev environment.

---

## Terraform Setup

### Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- Google provider `hashicorp/google` ~> 5.0
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
project_id = "your-gcp-project-id"
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
| `project_id` | string | (required) | GCP project ID |
| `region` | string | (required) | GCP region |
| `app_name` | string | (required) | Application name |
| `environment` | string | (required) | Deployment environment (dev, staging, prod) |
| `labels` | map(string) | `{}` | Labels to apply to the service |
| `service_name_override` | string | `""` | Override the service name (default: `app_name-environment`) |
| `container_image` | string | `""` | Full image URL (empty = Artifact Registry default) |
| `container_image_name` | string | `""` | Override image name in default URL (default: `app_name`) |
| `registry_repository_id` | string | (required) | Artifact Registry repo ID for default image URL |
| `cpu` | string | `1000m` | CPU allocation (1000m = 1 vCPU) |
| `memory` | string | `512Mi` | Memory allocation |
| `container_port` | number | `8080` | Port the container listens on |
| `min_instances` | number | `0` | Min instances (0 = scale-to-zero) |
| `max_instances` | number | `2` | Maximum instances |
| `service_account_email` | string | (required) | Service account email for Cloud Run |
| `extra_env_vars` | map(string) | `{}` | Additional environment variables |
| `startup_probe_timeout` | number | `240` | Startup probe timeout in seconds |
| `allow_unauthenticated` | bool | `true` | Public access via allUsers |

### Module: storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | (required) | GCP project ID |
| `region` | string | (required) | GCP region |
| `app_name` | string | (required) | Application name |
| `environment` | string | (required) | Deployment environment |
| `bucket_name_override` | string | `""` | Override bucket name (default: `project_id-app_name-environment`) |
| `labels` | map(string) | `{}` | Labels to apply |
| `force_destroy` | bool | `false` | Allow bucket deletion with objects |
| `versioning_enabled` | bool | `false` | Enable object versioning |
| `public_access_prevention` | string | `"inherited"` | Public access prevention mode |
| `temp_file_retention_days` | number | `90` | Days to retain temp files before deletion |
| `temp_file_prefixes` | list(string) | `["tmp/", "temp/", "cache/"]` | Prefixes considered temporary |
| `noncurrent_version_retention_days` | number | `30` | Days to retain noncurrent versions |

### Module: registry

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | (required) | GCP project ID |
| `region` | string | (required) | GCP region |
| `app_name` | string | (required) | Application name (used as repository ID) |
| `labels` | map(string) | `{}` | Labels to apply |
| `keep_count` | number | `10` | Recent tagged images to keep |
| `untagged_max_age_seconds` | number | `604800` | Max untagged image age in seconds (7d) |

### Module: monitoring

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | (required) | GCP project ID |
| `app_name` | string | (required) | Application name |
| `environment` | string | (required) | Deployment environment |
| `cloud_run_service_name` | string | (required) | Cloud Run service name to monitor |
| `cloud_run_service_url` | string | (required) | Cloud Run service URL for uptime checks |
| `region` | string | (required) | GCP region |
| `health_check_path` | string | `/health` | HTTP path for uptime checks |
| `uptime_check_period` | string | `60s` | Period between uptime checks |
| `alert_downtime_duration` | string | `60s` | Downtime duration before alerting |
| `error_rate_threshold` | number | `5` | 5xx error rate threshold for alerting |
| `notification_email` | string | `""` | Email for alert notifications |
| `enable_monitoring` | bool | `true` | Enable/disable all monitoring resources |

### Module: iam

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `project_id` | string | (required) | GCP project ID |
| `app_name` | string | (required) | Application name |
| `environment` | string | (required) | Deployment environment |
| `sa_account_id_override` | string | `""` | Override SA account ID (default: `app_name-environment`) |
| `additional_roles` | list(string) | `[]` | Additional IAM roles to grant |

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

Provisioned by the IAM module with these base roles:

| Role | Purpose |
|------|---------|
| `roles/storage.objectAdmin` | Read model checkpoints and data |
| `roles/artifactregistry.reader` | Pull container images |
| `roles/logging.logWriter` | Application logging |
| `roles/monitoring.metricWriter` | Metrics export |

**Note:** Production currently uses the default Compute Engine service account rather than the Terraform-provisioned SA. See the [Cloud Run Service](#cloud-run-service) section for details.

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

**Estimated monthly cost (low traffic):** $0 - $2

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

The Terraform configuration includes a budget alert (default: 10 EUR/month for dev, 50 EUR/month for prod) with notifications at 50%, 80%, and 100% thresholds. Budget resources are only created when the `billing_account` variable is provided.

### Total Estimated Monthly Cost

| Scenario | Estimate |
|----------|----------|
| Development (occasional training, low traffic) | $1 - $5 |
| Active development (weekly training, moderate traffic) | $5 - $15 |

---

## Remote State

Terraform state can be stored locally or in a GCS bucket for team collaboration. Each environment manages its own backend configuration independently.

**Current status:**
- **Prod:** Remote state is active, stored in `gs://ai-product-detector-487013-tfstate` with prefix `terraform/state/prod`.
- **Dev:** Local state by default. GCS backend block is present but commented out.

The GCS tfstate bucket is visible in the GCP Console (see `images/gcs-tfstate-bucket.jpg`).

### Setup (one-time, for new environments)

```bash
PROJECT_ID="<YOUR-PROJECT-ID>"

# Create state bucket
gsutil mb -l europe-west1 gs://${PROJECT_ID}-tfstate
gsutil versioning set on gs://${PROJECT_ID}-tfstate
```

### Enable for Dev

Uncomment the `backend "gcs"` block in `environments/dev/main.tf`:

```hcl
terraform {
  backend "gcs" {
    bucket = "<YOUR-PROJECT-ID>-tfstate"
    prefix = "terraform/state/dev"
  }
}
```

Then migrate:

```bash
terraform init -migrate-state
```

### State Isolation

Each environment uses a separate prefix within the same bucket to prevent state conflicts:

| Environment | Prefix |
|-------------|--------|
| dev | `terraform/state/dev` |
| prod | `terraform/state/prod` |

---

## Teardown

### Destroy all Terraform-managed resources

```bash
cd terraform/environments/dev  # or prod
terraform destroy
```

Type `yes` when prompted. This removes:
- Cloud Run service and IAM bindings
- Artifact Registry repository (and all images)
- IAM service account and role bindings
- Monitoring resources (uptime checks, alerts, notification channels)
- Billing budget alert (if created)
- Custom domain mapping (if created, prod only)

**Note:** The GCS bucket has `force_destroy = false` in production, meaning Terraform will refuse to delete it if it contains objects. To force deletion:

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
  <SA-ACCOUNT-ID>@<YOUR-PROJECT-ID>.iam.gserviceaccount.com
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
