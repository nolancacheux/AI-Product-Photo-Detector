# ---------------------------------------------------------------------------
# Monitoring module — Outputs
# ---------------------------------------------------------------------------

output "uptime_check_id" {
  description = "Uptime check ID"
  value       = var.enable_monitoring ? google_monitoring_uptime_check_config.health[0].uptime_check_id : null
}

output "uptime_alert_policy_name" {
  description = "Uptime alert policy resource name"
  value       = var.enable_monitoring ? google_monitoring_alert_policy.uptime_alert[0].name : null
}

output "error_rate_alert_policy_name" {
  description = "Error rate alert policy resource name"
  value       = var.enable_monitoring ? google_monitoring_alert_policy.error_rate_alert[0].name : null
}

output "notification_channel_name" {
  description = "Email notification channel resource name"
  value       = var.enable_monitoring && var.notification_email != "" ? google_monitoring_notification_channel.email[0].name : null
}
