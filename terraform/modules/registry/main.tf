# ---------------------------------------------------------------------------
# Registry module — Artifact Registry for Docker images
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.app_name
  description   = "Docker images for ${var.app_name}"
  format        = "DOCKER"
  labels        = var.labels

  cleanup_policy_dry_run = false

  # Keep the N most recent tagged images
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"

    most_recent_versions {
      keep_count = var.keep_count
    }
  }

  # Delete untagged images older than the threshold
  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"

    condition {
      tag_state  = "UNTAGGED"
      older_than = "${var.untagged_max_age_seconds}s"
    }
  }
}
