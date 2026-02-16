# ---------------------------------------------------------------------------
# Cloud Run module — FastAPI inference API service
# ---------------------------------------------------------------------------

locals {
  service_name  = var.service_name_override != "" ? var.service_name_override : "${var.app_name}-${var.environment}"
  image_name    = var.container_image_name != "" ? var.container_image_name : var.app_name
  default_image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.registry_repository_id}/${local.image_name}:latest"
  image         = var.container_image != "" ? var.container_image : local.default_image

  all_env = var.extra_env_vars
}

resource "google_cloud_run_v2_service" "api" {
  name     = local.service_name
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
        tcp_socket {
          port = var.container_port
        }
        timeout_seconds   = var.startup_probe_timeout
        period_seconds    = var.startup_probe_timeout
        failure_threshold = 1
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
