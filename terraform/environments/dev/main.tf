# ---------------------------------------------------------------------------
# DEV environment — Scale-to-zero, minimal resources, low cost
# ---------------------------------------------------------------------------

# Uncomment after creating the GCS bucket for remote state:
#   gsutil mb -l europe-west1 gs://<PROJECT_ID>-tfstate
#   gsutil versioning set on gs://<PROJECT_ID>-tfstate
#
# terraform {
#   backend "gcs" {
#     bucket = "<PROJECT_ID>-tfstate"
#     prefix = "terraform/state/dev"
#   }
# }

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  environment = "dev"
  labels = {
    app         = var.app_name
    environment = local.environment
    managed_by  = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Enable required GCP APIs
# ---------------------------------------------------------------------------

resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "billingbudgets.googleapis.com",
    "monitoring.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

module "iam" {
  source = "../../modules/iam"

  project_id  = var.project_id
  app_name    = var.app_name
  environment = local.environment

  depends_on = [google_project_service.required_apis]
}

module "storage" {
  source = "../../modules/storage"

  project_id    = var.project_id
  region        = var.region
  app_name      = var.app_name
  environment   = local.environment
  labels        = local.labels
  force_destroy = true # Safe to destroy in dev

  depends_on = [google_project_service.required_apis]
}

module "registry" {
  source = "../../modules/registry"

  project_id = var.project_id
  region     = var.region
  app_name   = var.app_name
  labels     = local.labels

  # More aggressive cleanup in dev
  keep_count               = 5
  untagged_max_age_seconds = 259200 # 3 days

  depends_on = [google_project_service.required_apis]
}

module "cloud_run" {
  source = "../../modules/cloud-run"

  project_id             = var.project_id
  region                 = var.region
  app_name               = var.app_name
  environment            = local.environment
  labels                 = local.labels
  container_image        = var.cloud_run_container_image
  registry_repository_id = module.registry.repository_id
  service_account_email  = module.iam.service_account_email
  gcs_bucket_name        = module.storage.bucket_name

  # Dev: scale-to-zero, minimal resources
  cpu           = "1000m"
  memory        = "512Mi"
  min_instances = 0
  max_instances = 2

  allow_unauthenticated = true

  depends_on = [google_project_service.required_apis]
}

module "monitoring" {
  source = "../../modules/monitoring"

  project_id             = var.project_id
  app_name               = var.app_name
  environment            = local.environment
  region                 = var.region
  cloud_run_service_name = module.cloud_run.service_name
  cloud_run_service_url  = module.cloud_run.service_url
  notification_email     = var.notification_email

  # Monitoring is optional in dev (default: disabled to save cost)
  enable_monitoring = var.enable_monitoring

  depends_on = [google_project_service.required_apis]
}

# ---------------------------------------------------------------------------
# Budget alert — 10€ for dev (critical for students!)
# ---------------------------------------------------------------------------

data "google_billing_account" "account" {
  count           = var.billing_account != "" ? 1 : 0
  billing_account = var.billing_account
}

resource "google_billing_budget" "monthly_budget" {
  count = var.billing_account != "" ? 1 : 0

  billing_account = var.billing_account
  display_name    = "${var.app_name}-${local.environment}-monthly-budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "EUR"
      units         = var.budget_amount
    }
  }

  threshold_rules {
    threshold_percent = 0.5
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 0.8
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "CURRENT_SPEND"
  }

  depends_on = [google_project_service.required_apis]
}
