# ---------------------------------------------------------------------------
# Backend — Remote state storage (recommended)
# ---------------------------------------------------------------------------
# Uncomment after creating the bucket manually:
#   gsutil mb -l europe-west1 gs://YOUR_PROJECT_ID-tfstate
#   gsutil versioning set on gs://YOUR_PROJECT_ID-tfstate
#
# terraform {
#   backend "gcs" {
#     bucket = "YOUR_PROJECT_ID-tfstate"
#     prefix = "terraform/state"
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
  bucket_name     = "${var.project_id}-mlops-data"
  registry_name   = replace(var.app_name, "-", "")
  service_account = "${var.app_name}-sa"
  labels = {
    app         = var.app_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---------------------------------------------------------------------------
# Enable required APIs
# ---------------------------------------------------------------------------
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "billingbudgets.googleapis.com",
  ])

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# GCS Bucket — DVC data & model storage
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "mlops_data" {
  name     = local.bucket_name
  location = var.region
  labels   = local.labels

  uniform_bucket_level_access = true
  force_destroy               = false
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 5
    }
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 90
      with_state = "ARCHIVED"
    }
  }

  depends_on = [google_project_service.required_apis]
}

# ---------------------------------------------------------------------------
# Artifact Registry — Docker images
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.app_name
  description   = "Docker images for ${var.app_name}"
  format        = "DOCKER"
  labels        = local.labels

  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"

    most_recent_versions {
      keep_count = 10
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"

    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  depends_on = [google_project_service.required_apis]
}

# ---------------------------------------------------------------------------
# Service Account — least-privilege identity for Cloud Run
# ---------------------------------------------------------------------------
resource "google_service_account" "app_sa" {
  account_id   = local.service_account
  display_name = "Service account for ${var.app_name}"
  description  = "Managed by Terraform — used by Cloud Run and CI/CD"

  depends_on = [google_project_service.required_apis]
}

resource "google_project_iam_member" "sa_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

# ---------------------------------------------------------------------------
# Cloud Run Service — FastAPI inference API
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "api" {
  name     = var.app_name
  location = var.region
  labels   = local.labels

  template {
    service_account = google_service_account.app_sa.email

    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      image = var.cloud_run_container_image != "" ? var.cloud_run_container_image : "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/${var.app_name}:latest"

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.mlops_data.name
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.docker_repo,
  ]
}

# Allow unauthenticated access to the API
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Budget Alert — avoid surprise bills (critical for students!)
# ---------------------------------------------------------------------------
data "google_billing_account" "account" {
  count          = var.billing_account != "" ? 1 : 0
  billing_account = var.billing_account
}

resource "google_billing_budget" "monthly_budget" {
  count = var.billing_account != "" ? 1 : 0

  billing_account = var.billing_account
  display_name    = "${var.app_name}-monthly-budget"

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
