# Terraform — GCP Infrastructure

Infrastructure as Code for the AI Product Photo Detector MLOps pipeline on Google Cloud Platform.

## Architecture Overview

| Resource | Purpose |
|---|---|
| **GCS Bucket** | DVC remote storage for datasets and model artifacts |
| **Artifact Registry** | Docker image repository (with auto-cleanup policy) |
| **Cloud Run** | Serverless deployment of the FastAPI inference API |
| **Service Account** | Least-privilege identity for Cloud Run |
| **Budget Alert** | Monthly spending alert to avoid surprise bills |

## Prerequisites

Before starting, make sure you have:

| Tool | Version | Install |
|---|---|---|
| **Google Cloud SDK** | Latest | [Install gcloud](https://cloud.google.com/sdk/docs/install) |
| **Terraform** | ≥ 1.5.0 | [Install Terraform](https://developer.hashicorp.com/terraform/install) |
| **GCP Billing Account** | Active | [Set up billing](https://cloud.google.com/billing/docs/how-to/create-billing-account) |

Verify your installations:

```bash
gcloud version
terraform version
```

## Step-by-Step Setup (New User)

### Step 1 — Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
```

### Step 2 — Create a GCP Project (if you don't have one)

```bash
# Create a new project
gcloud projects create my-mlops-project --name="MLOps Project"

# Set it as default
gcloud config set project my-mlops-project

# Link billing account (required for resource creation)
# Find your billing account ID:
gcloud billing accounts list

# Link it:
gcloud billing projects link my-mlops-project \
  --billing-account=XXXXXX-XXXXXX-XXXXXX
```

### Step 3 — Enable Required APIs

Terraform will enable APIs automatically, but you can also do it manually if needed:

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  billingbudgets.googleapis.com
```

### Step 4 — Create a Service Account for Terraform (optional but recommended)

> For personal/student projects, using `gcloud auth application-default login` is sufficient.
> For shared projects or CI/CD, create a dedicated service account:

```bash
# Create the service account
gcloud iam service-accounts create terraform-admin \
  --display-name="Terraform Admin"

# Grant necessary permissions
gcloud projects add-iam-policy-binding my-mlops-project \
  --member="serviceAccount:terraform-admin@my-mlops-project.iam.gserviceaccount.com" \
  --role="roles/editor"

gcloud projects add-iam-policy-binding my-mlops-project \
  --member="serviceAccount:terraform-admin@my-mlops-project.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountAdmin"

# Download key file
gcloud iam service-accounts keys create terraform-key.json \
  --iam-account=terraform-admin@my-mlops-project.iam.gserviceaccount.com

# Set the credential env variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/terraform-key.json"
```

> ⚠️ **Never commit `terraform-key.json`!** It's already in `.gitignore`.

### Step 5 — Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and set **at minimum**:

```hcl
project_id      = "my-mlops-project"       # REQUIRED
billing_account = "XXXXXX-XXXXXX-XXXXXX"    # Recommended (for budget alerts)
```

See [terraform.tfvars.example](terraform.tfvars.example) for all available variables.

### Step 6 — Deploy Infrastructure

```bash
# Initialize Terraform (downloads providers)
terraform init

# Preview what will be created
terraform plan

# Apply (creates all resources)
terraform apply
```

Type `yes` when prompted.

### Step 7 — Verify Outputs

```bash
terraform output
```

Expected outputs:

| Output | Description |
|---|---|
| `cloud_run_url` | Your API endpoint |
| `gcs_bucket_name` | DVC remote bucket name |
| `gcs_bucket_url` | Full GCS URL for DVC config |
| `artifact_registry_url` | Docker push target |
| `service_account_email` | Service account for CI/CD |
| `docker_push_command` | Ready-to-use push command |

## Remote State (Optional but Recommended)

For team collaboration or CI/CD, store state in a GCS bucket:

```bash
# Create the state bucket (one-time, before terraform init)
gsutil mb -l europe-west1 gs://YOUR_PROJECT_ID-tfstate
gsutil versioning set on gs://YOUR_PROJECT_ID-tfstate
```

Then uncomment the backend block in `main.tf`:

```hcl
terraform {
  backend "gcs" {
    bucket = "YOUR_PROJECT_ID-tfstate"
    prefix = "terraform/state"
  }
}
```

Re-initialize to migrate:

```bash
terraform init -migrate-state
```

## Post-Deployment Usage

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
  --image $(terraform output -raw artifact_registry_url)/ai-product-detector:latest \
  --region $(terraform output -raw region)
```

## Cost Management

This configuration is optimized for student projects:

| Setting | Value | Why |
|---|---|---|
| `cloud_run_min_instances` | `0` | Scale-to-zero when idle (no idle cost) |
| `cloud_run_max_instances` | `2` | Prevents runaway scaling |
| `cloud_run_cpu` | `1000m` | 1 vCPU (minimal) |
| `cloud_run_memory` | `512Mi` | Sufficient for inference |
| `budget_amount` | `10€` | Alert before spending too much |
| GCS lifecycle rules | 5 versions, 90-day archive cleanup | Prevents storage bloat |
| AR cleanup policy | Keep 10 latest, delete untagged after 7d | Saves registry storage |

> 💡 **Estimated cost**: Under free tier for light usage. Budget alert at 10€/month by default.

## Destroy

To tear down all resources:

```bash
terraform destroy
```

> ⚠️ The GCS bucket has `force_destroy = false`. Empty it first or temporarily set `force_destroy = true` in `main.tf` before destroying.

## File Structure

```
terraform/
├── main.tf                  # Provider, resources (GCS, AR, Cloud Run, IAM, Budget)
├── variables.tf             # Input variables with validation & defaults
├── outputs.tf               # Useful outputs for downstream tools
├── terraform.tfvars.example # Template — copy to terraform.tfvars
├── .gitignore               # Excludes state, keys, and tfvars
└── README.md                # This file
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `Error 403: billing account not found` | Check `billing_account` value or leave empty to skip budget |
| `API not enabled` | Run `gcloud services enable <api>` or wait for Terraform to enable it |
| `Permission denied` | Make sure `gcloud auth application-default login` is done |
| `Cloud Run: container failed to start` | Push a valid Docker image first, then re-apply |
| `Bucket already exists` | GCS bucket names are global — change `project_id` or `app_name` |
