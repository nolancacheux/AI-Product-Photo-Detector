"""Structured logging configuration with request tracing.

Outputs JSON logs compatible with Google Cloud Run / Cloud Logging.
Maps structlog levels to Cloud Run's `severity` field for proper log level filtering.
"""

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

# Context variable for request ID tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Cloud Run severity mapping
_CLOUD_RUN_SEVERITY = {
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARNING",
    "error": "ERROR",
    "critical": "CRITICAL",
}


def get_request_id() -> str | None:
    """Get the current request ID."""
    return request_id_var.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set the current request ID.

    Args:
        request_id: Request ID to set. If None, generates a new one.

    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the current request ID."""
    request_id_var.set(None)


def add_request_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add request ID to log events."""
    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_service_info(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service information to log events."""
    event_dict["service"] = os.getenv("SERVICE_NAME", "ai-photo-detector")
    event_dict["version"] = os.getenv("SERVICE_VERSION", "1.0.0")
    return event_dict


def add_cloud_run_severity(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Map log level to Cloud Run severity field.

    Cloud Run / Cloud Logging expects a top-level `severity` field
    to properly categorize log entries. Without this, all structured
    logs appear as DEFAULT severity.
    """
    level = event_dict.get("level", "info")
    event_dict["severity"] = _CLOUD_RUN_SEVERITY.get(level, "DEFAULT")
    return event_dict


def add_cloud_trace_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add Cloud Trace context for request correlation.

    Cloud Run sets X-Cloud-Trace-Context header. If the trace_id
    is available in the context, format it for Cloud Logging.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    trace_id = event_dict.pop("trace_id", None)
    if project_id and trace_id:
        event_dict["logging.googleapis.com/trace"] = (
            f"projects/{project_id}/traces/{trace_id}"
        )
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    include_service_info: bool = True,
) -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output logs in JSON format (production).
        include_service_info: If True, add service name/version to logs.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        force=True,
    )

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_request_id,
        structlog.stdlib.ExtraAdder(),
    ]

    if include_service_info:
        shared_processors.append(add_service_info)

    if json_format:
        # Cloud Run compatible: add severity + trace context
        shared_processors.append(add_cloud_run_severity)
        shared_processors.append(add_cloud_trace_context)
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)
