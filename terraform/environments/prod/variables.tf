# ---------------------------------------------------------------------------
# PROD environment — Variables
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

# --- Custom domain ---

variable "custom_domain" {
  description = "Custom domain for Cloud Run (leave empty to skip domain mapping)"
  type        = string
  default     = ""
}

# --- IAM ---

variable "additional_iam_roles" {
  description = "Additional IAM roles for the prod service account"
  type        = list(string)
  default     = []
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
  default     = 50
}

# --- Monitoring ---

variable "notification_email" {
  description = "Email address for alert notifications (recommended for prod)"
  type        = string
  default     = ""
}
