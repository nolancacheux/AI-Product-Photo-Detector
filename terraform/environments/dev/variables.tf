# ---------------------------------------------------------------------------
# DEV environment — Variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string

  validation {
    condition     = length(var.project_id) > 0
    error_message = "project_id must not be empty."
  }
}

variable "region" {
  description = "GCP region for resource deployment"
  type        = string
  default     = "europe-west1"
}

variable "app_name" {
  description = "Application name used for resource naming"
  type        = string
  default     = "ai-product-detector"
}

# --- Cloud Run ---

variable "cloud_run_container_image" {
  description = "Container image to deploy (leave empty to use Artifact Registry default)"
  type        = string
  default     = ""
}

# --- Budget ---

variable "billing_account" {
  description = "GCP billing account ID (format: XXXXXX-XXXXXX-XXXXXX). Leave empty to skip budget."
  type        = string
  default     = ""
}

variable "budget_amount" {
  description = "Monthly budget alert threshold in EUR"
  type        = number
  default     = 10
}

# --- Monitoring ---

variable "notification_email" {
  description = "Email address for alert notifications (leave empty to skip)"
  type        = string
  default     = ""
}

variable "enable_monitoring" {
  description = "Enable monitoring resources (uptime checks, alerts)"
  type        = bool
  default     = false
}
