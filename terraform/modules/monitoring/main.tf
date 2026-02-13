# ---------------------------------------------------------------------------
# Monitoring module — Uptime checks, alert policies, notifications
# ---------------------------------------------------------------------------

locals {
  # Extract hostname from Cloud Run URL (remove https:// prefix)
  service_host = replace(var.cloud_run_service_url, "https://", "")
  display_name = "${var.app_name}-${var.environment}"
}

# ---------------------------------------------------------------------------
# Notification channel (email)
# ---------------------------------------------------------------------------

resource "google_monitoring_notification_channel" "email" {
  count = var.enable_monitoring && var.notification_email != "" ? 1 : 0

  display_name = "${local.display_name}-email"
  type         = "email"

  labels = {
    email_address = var.notification_email
  }

  force_delete = false
}

locals {
  notification_channels = var.enable_monitoring && var.notification_email != "" ? [google_monitoring_notification_channel.email[0].name] : []
}

# ---------------------------------------------------------------------------
# Uptime check — HTTP GET /health
# ---------------------------------------------------------------------------

resource "google_monitoring_uptime_check_config" "health" {
  count = var.enable_monitoring ? 1 : 0

  display_name = "${local.display_name}-health-check"
  timeout      = "10s"
  period       = var.uptime_check_period
  project      = var.project_id

  http_check {
    path         = var.health_check_path
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = local.service_host
    }
  }
}

# ---------------------------------------------------------------------------
# Alert policy — Service down for > threshold
# ---------------------------------------------------------------------------

resource "google_monitoring_alert_policy" "uptime_alert" {
  count = var.enable_monitoring ? 1 : 0

  display_name = "${local.display_name}-uptime-alert"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Uptime check failure"

    condition_threshold {
      filter          = "resource.type = \"uptime_url\" AND metric.type = \"monitoring.googleapis.com/uptime_check/check_passed\" AND metric.labels.check_id = \"${google_monitoring_uptime_check_config.health[0].uptime_check_id}\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1
      duration        = var.alert_downtime_duration

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
        group_by_fields      = ["resource.label.project_id"]
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = local.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "The service ${local.display_name} health check has been failing. Check Cloud Run logs: https://console.cloud.google.com/run/detail/${var.region}/${var.cloud_run_service_name}/logs?project=${var.project_id}"
    mime_type = "text/markdown"
  }
}

# ---------------------------------------------------------------------------
# Alert policy — Error rate > threshold (5xx responses)
# ---------------------------------------------------------------------------

resource "google_monitoring_alert_policy" "error_rate_alert" {
  count = var.enable_monitoring ? 1 : 0

  display_name = "${local.display_name}-error-rate-alert"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run 5xx error rate > ${var.error_rate_threshold}%"

    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.cloud_run_service_name}\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.error_rate_threshold
      duration        = "60s"

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = local.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "The service ${local.display_name} is experiencing a high error rate (>${var.error_rate_threshold}%). Check Cloud Run logs: https://console.cloud.google.com/run/detail/${var.region}/${var.cloud_run_service_name}/logs?project=${var.project_id}"
    mime_type = "text/markdown"
  }
}
