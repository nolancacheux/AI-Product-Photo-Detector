"""Structured logging configuration."""

import logging
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output logs in JSON format.
    """
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

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
