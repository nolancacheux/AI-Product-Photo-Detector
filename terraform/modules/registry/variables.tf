# ---------------------------------------------------------------------------
# Registry module — Input variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the registry"
  type        = string
}

variable "app_name" {
  description = "Application name used as repository ID"
  type        = string
}

variable "labels" {
  description = "Labels to apply to the repository"
  type        = map(string)
  default     = {}
}

variable "keep_count" {
  description = "Number of recent tagged images to keep"
  type        = number
  default     = 10
}

variable "untagged_max_age_seconds" {
  description = "Max age in seconds for untagged images before deletion"
  type        = number
  default     = 604800 # 7 days
}
