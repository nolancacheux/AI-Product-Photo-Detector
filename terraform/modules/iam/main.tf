# ---------------------------------------------------------------------------
# IAM module — Service accounts and role bindings
# ---------------------------------------------------------------------------

locals {
  sa_account_id = "${var.app_name}-${var.environment}"

  # Base roles required by Cloud Run service
  base_roles = [
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ]

  all_roles = concat(local.base_roles, var.additional_roles)
}

resource "google_service_account" "app" {
  account_id   = local.sa_account_id
  display_name = "Service account for ${var.app_name} (${var.environment})"
  description  = "Managed by Terraform — used by Cloud Run and CI/CD"
}

resource "google_project_iam_member" "sa_roles" {
  for_each = toset(local.all_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.app.email}"
}
