# ---------------------------------------------------------------------------
# Storage module — GCS bucket for DVC data & model storage
# ---------------------------------------------------------------------------

locals {
  bucket_name = "${var.project_id}-${var.app_name}-${var.environment}"
}

resource "google_storage_bucket" "mlops_data" {
  name     = local.bucket_name
  location = var.region
  labels   = var.labels

  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  # Keep only N most recent versions of each object
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = var.versioning_max_versions
    }
  }

  # Delete archived objects after retention period
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age        = var.archive_retention_days
      with_state = "ARCHIVED"
    }
  }
}
