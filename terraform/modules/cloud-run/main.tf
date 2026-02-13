# ---------------------------------------------------------------------------
# Cloud Run module — FastAPI inference API service
# ---------------------------------------------------------------------------

locals {
  default_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.registry_repository_id}/${var.app_name}:latest"
  image         = var.container_image != "" ? var.container_image : local.default_image

  # Merge base env vars with extra env vars
  base_env = {
    ENVIRONMENT = var.environment
    GCS_BUCKET  = var.gcs_bucket_name
  }
  all_env = merge(local.base_env, var.extra_env_vars)
}

resource "google_cloud_run_v2_service" "api" {
  name     = "${var.app_name}-${var.environment}"
  location = var.region
  labels   = var.labels

  template {
    service_account = var.service_account_email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = local.image

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      ports {
        container_port = var.container_port
      }

      # Inject all environment variables dynamically
      dynamic "env" {
        for_each = local.all_env
        content {
          name  = env.key
          value = env.value
        }
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
}

# Allow unauthenticated access (public API)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count = var.allow_unauthenticated ? 1 : 0

  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
