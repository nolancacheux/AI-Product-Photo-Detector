# ---------------------------------------------------------------------------
# PROD environment — Always-on, higher resources, custom domain ready
# ---------------------------------------------------------------------------

# Uncomment after creating the GCS bucket for remote state:
#   gsutil mb -l europe-west1 gs://<PROJECT_ID>-tfstate
#   gsutil versioning set on gs://<PROJECT_ID>-tfstate
#
# terraform {
#   backend "gcs" {
#     bucket = "<PROJECT_ID>-tfstate"
#     prefix = "terraform/state/prod"
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
  environment = "prod"
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

  # Prod may need additional roles (e.g., for custom domain mapping)
  additional_roles = var.additional_iam_roles

  depends_on = [google_project_service.required_apis]
}

module "storage" {
  source = "../../modules/storage"

  project_id    = var.project_id
  region        = var.region
  app_name      = var.app_name
  environment   = local.environment
  labels        = local.labels
  force_destroy = false # Never allow accidental data loss in prod

  # Keep more versions in prod
  versioning_max_versions = 10
  archive_retention_days  = 180

  depends_on = [google_project_service.required_apis]
}

module "registry" {
  source = "../../modules/registry"

  project_id = var.project_id
  region     = var.region
  app_name   = var.app_name
  labels     = local.labels

  # Keep more images in prod
  keep_count               = 20
  untagged_max_age_seconds = 1209600 # 14 days

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

  # Prod: always-on, higher resources
  cpu           = "1000m"
  memory        = "1Gi"
  min_instances = 1
  max_instances = 10

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

  # Monitoring is always enabled in prod
  enable_monitoring       = true
  alert_downtime_duration = "60s"
  error_rate_threshold    = 5

  depends_on = [google_project_service.required_apis]
}

# ---------------------------------------------------------------------------
# Custom domain mapping (optional)
# ---------------------------------------------------------------------------

resource "google_cloud_run_domain_mapping" "custom_domain" {
  count = var.custom_domain != "" ? 1 : 0

  location = var.region
  name     = var.custom_domain

  metadata {
    namespace = var.project_id
    labels    = local.labels
  }

  spec {
    route_name = module.cloud_run.service_name
  }

  depends_on = [module.cloud_run]
}

# ---------------------------------------------------------------------------
# Budget alert — 50€ for prod
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
