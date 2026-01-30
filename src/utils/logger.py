"""Structured logging configuration with request tracing."""

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

# Context variable for request ID tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


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
    """Add request ID to log events.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary.

    Returns:
        Updated event dictionary.
    """
    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_service_info(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service information to log events.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary.

    Returns:
        Updated event dictionary.
    """
    event_dict["service"] = os.getenv("SERVICE_NAME", "ai-photo-detector")
    event_dict["version"] = os.getenv("SERVICE_VERSION", "1.0.0")
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    include_service_info: bool = True,
) -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output logs in JSON format.
        include_service_info: If True, add service name/version to logs.
    """
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_request_id,
        structlog.stdlib.ExtraAdder(),
    ]

    if include_service_info:
        shared_processors.insert(4, add_service_info)

    if json_format:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
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


class LoggingMiddleware:
    """ASGI middleware for request logging with tracing."""

    def __init__(self, app, logger: structlog.BoundLogger | None = None) -> None:
        """Initialize middleware.

        Args:
            app: ASGI application.
            logger: Logger instance to use.
        """
        self.app = app
        self.logger = logger or get_logger("http")

    async def __call__(self, scope, receive, send) -> None:
        """Process request with logging."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID
        request_id = None
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == b"x-request-id":
                request_id = header_value.decode()
                break

        request_id = set_request_id(request_id)

        # Extract request info
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        client = scope.get("client", ("unknown", 0))

        import time
        start_time = time.time()

        # Track response status
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Add request ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Log request completion
            log_method = self.logger.info if status_code < 400 else self.logger.warning
            if status_code >= 500:
                log_method = self.logger.error

            log_method(
                "HTTP request",
                method=method,
                path=path,
                status=status_code,
                duration_ms=round(duration_ms, 2),
                client_ip=client[0] if client else "unknown",
            )

            clear_request_id()
