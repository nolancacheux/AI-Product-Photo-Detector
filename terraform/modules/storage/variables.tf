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

variable "bucket_name_override" {
  description = "Override the bucket name (default: project_id-app_name-environment)"
  type        = string
  default     = ""
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

variable "versioning_enabled" {
  description = "Enable object versioning on the bucket"
  type        = bool
  default     = false
}

variable "public_access_prevention" {
  description = "Public access prevention mode (enforced or inherited)"
  type        = string
  default     = "inherited"
}

variable "temp_file_retention_days" {
  description = "Days to retain temporary files before deletion"
  type        = number
  default     = 90
}

variable "temp_file_prefixes" {
  description = "Object prefixes considered temporary (for lifecycle cleanup)"
  type        = list(string)
  default     = ["tmp/", "temp/", "cache/"]
}

variable "noncurrent_version_retention_days" {
  description = "Days to retain noncurrent object versions before deletion"
  type        = number
  default     = 30
}
