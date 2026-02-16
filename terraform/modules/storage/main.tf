# ---------------------------------------------------------------------------
# Storage module — GCS bucket for DVC data & model storage
# ---------------------------------------------------------------------------

locals {
  bucket_name = var.bucket_name_override != "" ? var.bucket_name_override : "${var.project_id}-${var.app_name}-${var.environment}"
}

resource "google_storage_bucket" "mlops_data" {
  name     = local.bucket_name
  location = var.region
  labels   = var.labels

  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy
  public_access_prevention    = var.public_access_prevention

  versioning {
    enabled = var.versioning_enabled
  }

  # Delete temporary files after retention period
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age            = var.temp_file_retention_days
      matches_prefix = var.temp_file_prefixes
    }
  }

  # Delete noncurrent versions after retention period
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      days_since_noncurrent_time = var.noncurrent_version_retention_days
      with_state                 = "ARCHIVED"
    }
  }
}
