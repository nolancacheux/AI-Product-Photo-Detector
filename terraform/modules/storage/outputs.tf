# ---------------------------------------------------------------------------
# Storage module — Outputs
# ---------------------------------------------------------------------------

output "bucket_name" {
  description = "GCS bucket name"
  value       = google_storage_bucket.mlops_data.name
}

output "bucket_url" {
  description = "GCS bucket URL (gs:// format, use for DVC remote)"
  value       = "gs://${google_storage_bucket.mlops_data.name}"
}

output "bucket_self_link" {
  description = "GCS bucket self link"
  value       = google_storage_bucket.mlops_data.self_link
}
