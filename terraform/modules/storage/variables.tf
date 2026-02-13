# ---------------------------------------------------------------------------
# Storage module — Input variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for bucket location"
  type        = string
}

variable "app_name" {
  description = "Application name used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "force_destroy" {
  description = "Allow bucket deletion even if it contains objects (use true only in dev)"
  type        = bool
  default     = false
}

variable "versioning_max_versions" {
  description = "Number of newer object versions to keep before deleting old ones"
  type        = number
  default     = 5
}

variable "archive_retention_days" {
  description = "Days to retain archived objects before deletion"
  type        = number
  default     = 90
}
