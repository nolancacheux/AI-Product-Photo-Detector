# ---------------------------------------------------------------------------
# PROD environment — Outputs
# ---------------------------------------------------------------------------

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "environment" {
  description = "Deployment environment"
  value       = "prod"
}

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = module.cloud_run.service_url
}

output "cloud_run_service_name" {
  description = "Cloud Run service name"
  value       = module.cloud_run.service_name
}

output "gcs_bucket_name" {
  description = "GCS bucket name for DVC data and model storage"
  value       = module.storage.bucket_name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL (use for DVC remote)"
  value       = module.storage.bucket_url
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker repository URL"
  value       = module.registry.repository_url
}

output "service_account_email" {
  description = "Service account email used by Cloud Run"
  value       = module.iam.service_account_email
}

output "docker_push_command" {
  description = "Example command to push a Docker image"
  value       = module.registry.docker_push_command
}

output "custom_domain" {
  description = "Custom domain (if configured)"
  value       = var.custom_domain != "" ? var.custom_domain : "not configured"
}

output "monitoring_uptime_check_id" {
  description = "Uptime check ID"
  value       = module.monitoring.uptime_check_id
}
