# ---------------------------------------------------------------------------
# Monitoring module — Input variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "app_name" {
  description = "Application name for display in alerts"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "cloud_run_service_name" {
  description = "Cloud Run service name to monitor"
  type        = string
}

variable "cloud_run_service_url" {
  description = "Cloud Run service URL for uptime checks"
  type        = string
}

variable "region" {
  description = "GCP region where Cloud Run is deployed"
  type        = string
}

variable "health_check_path" {
  description = "HTTP path for uptime health checks"
  type        = string
  default     = "/health"
}

variable "uptime_check_period" {
  description = "Period between uptime checks in seconds"
  type        = string
  default     = "60s"
}

variable "alert_downtime_duration" {
  description = "Duration of downtime before alerting (e.g. 60s)"
  type        = string
  default     = "60s"
}

variable "error_rate_threshold" {
  description = "Error rate percentage threshold for alerting (0-100)"
  type        = number
  default     = 5
}

variable "notification_email" {
  description = "Email address for alert notifications (leave empty to skip)"
  type        = string
  default     = ""
}

variable "enable_monitoring" {
  description = "Enable monitoring resources (set false to skip in dev)"
  type        = bool
  default     = true
}
