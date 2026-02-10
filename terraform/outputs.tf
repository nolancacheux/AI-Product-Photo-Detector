output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.api.uri
}

output "cloud_run_service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.api.name
}

output "gcs_bucket_name" {
  description = "GCS bucket name for DVC data and model storage"
  value       = google_storage_bucket.mlops_data.name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL (use for DVC remote)"
  value       = "gs://${google_storage_bucket.mlops_data.name}"
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "service_account_email" {
  description = "Service account email used by Cloud Run"
  value       = google_service_account.app_sa.email
}

output "docker_push_command" {
  description = "Example command to push a Docker image"
  value       = "docker push ${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/${var.app_name}:latest"
}
