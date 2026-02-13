# ---------------------------------------------------------------------------
# Cloud Run module — Input variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run deployment"
  type        = string
}

variable "app_name" {
  description = "Application / Cloud Run service name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "labels" {
  description = "Labels to apply to the service"
  type        = map(string)
  default     = {}
}

# --- Container configuration ---

variable "container_image" {
  description = "Full container image URL. If empty, defaults to Artifact Registry latest."
  type        = string
  default     = ""
}

variable "registry_repository_id" {
  description = "Artifact Registry repository ID (used to build default image URL)"
  type        = string
}

variable "cpu" {
  description = "CPU allocation (e.g. 1000m = 1 vCPU)"
  type        = string
  default     = "1000m"
}

variable "memory" {
  description = "Memory allocation (e.g. 512Mi, 1Gi)"
  type        = string
  default     = "512Mi"
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
  default     = 8080
}

# --- Scaling ---

variable "min_instances" {
  description = "Minimum number of instances (0 = scale-to-zero)"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 2
}

# --- Service account ---

variable "service_account_email" {
  description = "Service account email for Cloud Run"
  type        = string
}

# --- Environment variables ---

variable "gcs_bucket_name" {
  description = "GCS bucket name to inject as GCS_BUCKET env var"
  type        = string
}

variable "extra_env_vars" {
  description = "Additional environment variables for the container"
  type        = map(string)
  default     = {}
}

# --- Access ---

variable "allow_unauthenticated" {
  description = "Allow unauthenticated (public) access to the service"
  type        = bool
  default     = true
}
