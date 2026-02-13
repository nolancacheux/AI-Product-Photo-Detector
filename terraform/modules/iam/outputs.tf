# ---------------------------------------------------------------------------
# IAM module — Outputs
# ---------------------------------------------------------------------------

output "service_account_email" {
  description = "Service account email address"
  value       = google_service_account.app.email
}

output "service_account_id" {
  description = "Service account unique ID"
  value       = google_service_account.app.id
}

output "service_account_name" {
  description = "Service account fully-qualified name"
  value       = google_service_account.app.name
}
