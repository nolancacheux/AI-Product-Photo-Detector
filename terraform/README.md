# Terraform — AI Product Photo Detector Infrastructure

Production-grade, modular Terraform configuration for deploying the AI Product Photo Detector on Google Cloud Platform.

## Architecture

```
terraform/
├── environments/          # Per-environment configurations
│   ├── dev/               # Development (scale-to-zero, 512Mi, 10€ budget)
│   │   ├── main.tf        # Module calls with dev values
│   │   ├── variables.tf   # Dev-specific variables
│   │   ├── outputs.tf     # Dev outputs
│   │   └── terraform.tfvars
│   └── prod/              # Production (min 1 instance, 1Gi, 50€ budget)
│       ├── main.tf        # Module calls with prod values
│       ├── variables.tf   # Prod-specific variables
│       ├── outputs.tf     # Prod outputs
│       └── terraform.tfvars
├── modules/               # Reusable infrastructure modules
│   ├── cloud-run/         # Cloud Run service (FastAPI API)
│   ├── storage/           # GCS buckets (DVC data & models)
│   ├── registry/          # Artifact Registry (Docker images)
│   ├── monitoring/        # Uptime checks, alerts, notifications
│   └── iam/               # Service accounts & role bindings
├── backend.tf             # Remote state documentation
├── versions.tf            # Required providers
└── README.md              # This file
```

## Environment Comparison

| Setting               | Dev                    | Prod                        |
|-----------------------|------------------------|-----------------------------|
| Min instances         | 0 (scale-to-zero)     | 1 (always-on)               |
| Max instances         | 2                      | 10                          |
| Memory                | 512Mi                  | 1Gi                         |
| Budget                | 10€/month              | 50€/month                   |
| Monitoring            | Optional (off)         | Always enabled               |
| Bucket force_destroy  | true                   | false                       |
| Image retention       | 5 recent / 3d untagged | 20 recent / 14d untagged    |
| Custom domain         | N/A                    | Supported                   |

## Prerequisites

1. **Google Cloud SDK** installed and authenticated:
   ```bash
   gcloud auth application-default login
   ```

2. **Terraform** >= 1.5.0 installed:
   ```bash
   terraform version
   ```

3. **GCP Project** with billing enabled:
   ```bash
   gcloud projects list
   gcloud billing accounts list
   ```

## Quick Start

### 1. Configure your environment

```bash
# Choose your environment
cd terraform/environments/dev    # or /prod

# Edit terraform.tfvars with your project ID
vim terraform.tfvars
```

At minimum, set:
```hcl
project_id = "your-actual-gcp-project-id"
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan and review

```bash
terraform plan
```

### 4. Apply

```bash
terraform apply
```

### 5. Get outputs

```bash
# Service URL
terraform output cloud_run_url

# All outputs
terraform output
```

## Remote State (Recommended for Teams)

By default, state is stored locally. For team collaboration, enable GCS remote state:

### Setup (one-time)

```bash
# Replace with your project ID
PROJECT_ID="your-project-id"

# Create state bucket
gsutil mb -l europe-west1 gs://${PROJECT_ID}-tfstate
gsutil versioning set on gs://${PROJECT_ID}-tfstate
```

### Enable

Uncomment the `backend "gcs"` block in your environment's `main.tf`:

```hcl
terraform {
  backend "gcs" {
    bucket = "your-project-id-tfstate"
    prefix = "terraform/state/dev"  # or "terraform/state/prod"
  }
}
```

Then migrate:

```bash
terraform init -migrate-state
```

## Module Reference

### cloud-run

Deploys a Cloud Run v2 service with configurable scaling, resources, and health probes.

| Variable | Description | Default |
|----------|-------------|---------|
| `cpu` | CPU allocation | `1000m` |
| `memory` | Memory allocation | `512Mi` |
| `min_instances` | Min instances (0 = scale-to-zero) | `0` |
| `max_instances` | Max instances | `2` |
| `allow_unauthenticated` | Public access | `true` |

### storage

Creates a GCS bucket with versioning, lifecycle rules, and enforced public access prevention.

| Variable | Description | Default |
|----------|-------------|---------|
| `force_destroy` | Allow bucket deletion with objects | `false` |
| `versioning_max_versions` | Versions to keep | `5` |
| `archive_retention_days` | Archive retention | `90` |

### registry

Creates an Artifact Registry Docker repository with automatic cleanup policies.

| Variable | Description | Default |
|----------|-------------|---------|
| `keep_count` | Recent images to keep | `10` |
| `untagged_max_age_seconds` | Max untagged image age | `604800` (7d) |

### monitoring

Sets up uptime checks on `/health`, alert policies for downtime and error rate, and email notifications.

| Variable | Description | Default |
|----------|-------------|---------|
| `enable_monitoring` | Enable/disable all monitoring | `true` |
| `health_check_path` | HTTP path to check | `/health` |
| `alert_downtime_duration` | Downtime before alert | `60s` |
| `error_rate_threshold` | 5xx error rate % threshold | `5` |

### iam

Creates a dedicated service account with least-privilege roles for Cloud Run.

| Variable | Description | Default |
|----------|-------------|---------|
| `additional_roles` | Extra IAM roles | `[]` |

## Common Operations

### Deploy a new image

```bash
# Build and push
IMAGE="europe-west1-docker.pkg.dev/YOUR_PROJECT/ai-product-detector/ai-product-detector:v1.0.0"
docker build -t $IMAGE .
docker push $IMAGE

# Deploy via Terraform
cd terraform/environments/prod
terraform apply -var="cloud_run_container_image=$IMAGE"
```

### Destroy an environment

```bash
cd terraform/environments/dev
terraform destroy
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

## Cost Optimization

- **Dev**: Scale-to-zero means you only pay when the API receives traffic
- **Prod**: Min 1 instance keeps the service warm (no cold starts) but costs ~$5-15/month
- **Budget alerts** at 50%, 80%, and 100% of your limit
- **Artifact Registry cleanup** removes old images automatically
- **GCS lifecycle rules** delete old object versions

## Troubleshooting

### "Error creating Service: permission denied"
```bash
gcloud services enable run.googleapis.com --project=YOUR_PROJECT
```

### "Backend initialization required"
```bash
terraform init -reconfigure
```

### "Resource already exists"
Import the existing resource:
```bash
terraform import module.cloud_run.google_cloud_run_v2_service.api projects/PROJECT/locations/REGION/services/SERVICE
```
