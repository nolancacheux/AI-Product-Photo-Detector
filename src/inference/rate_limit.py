"""Rate limiter instance shared between api.py and route modules."""

import os

from fastapi import Request

from slowapi import Limiter


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP behind Cloud Run / reverse proxies.

    Cloud Run sets X-Forwarded-For with the real client IP as first entry.
    Falls back to request.client.host if header is absent.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


_enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"

limiter = Limiter(
    key_func=_get_real_client_ip,
    default_limits=["200/minute"],
    enabled=_enabled,
)
