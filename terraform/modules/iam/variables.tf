# ---------------------------------------------------------------------------
# IAM module — Input variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "app_name" {
  description = "Application name used for service account naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "additional_roles" {
  description = "Additional IAM roles to grant to the service account"
  type        = list(string)
  default     = []
}
