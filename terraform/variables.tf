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

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

# ---------------------------------------------------------------------------
# Cloud Run configuration
# ---------------------------------------------------------------------------

variable "cloud_run_container_image" {
  description = "Container image to deploy on Cloud Run (leave empty to use Artifact Registry default)"
  type        = string
  default     = ""
}

variable "cloud_run_cpu" {
  description = "CPU allocation for Cloud Run service (e.g. 1000m = 1 vCPU)"
  type        = string
  default     = "1000m"
}

variable "cloud_run_memory" {
  description = "Memory allocation for Cloud Run service"
  type        = string
  default     = "512Mi"
}

variable "cloud_run_max_instances" {
  description = "Maximum number of Cloud Run instances (keep low for student projects)"
  type        = number
  default     = 2

  validation {
    condition     = var.cloud_run_max_instances >= 1 && var.cloud_run_max_instances <= 10
    error_message = "cloud_run_max_instances must be between 1 and 10."
  }
}

variable "cloud_run_min_instances" {
  description = "Minimum number of Cloud Run instances (0 = scale to zero, saves money)"
  type        = number
  default     = 0

  validation {
    condition     = var.cloud_run_min_instances >= 0 && var.cloud_run_min_instances <= 3
    error_message = "cloud_run_min_instances must be between 0 and 3."
  }
}

# ---------------------------------------------------------------------------
# Budget & Billing
# ---------------------------------------------------------------------------

variable "billing_account" {
  description = "GCP billing account ID (format: XXXXXX-XXXXXX-XXXXXX). Leave empty to skip budget creation."
  type        = string
  default     = ""
}

variable "budget_amount" {
  description = "Monthly budget alert threshold in EUR"
  type        = number
  default     = 10
}
