"""Informational route handlers: /, /privacy."""

from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["Info"])


@router.get("/privacy")
async def privacy() -> dict[str, Any]:
    """Privacy policy summary.

    Returns:
        Privacy information about data handling.
    """
    return {
        "data_retention": "none",
        "image_storage": "Images are processed in-memory only and never saved to disk.",
        "logging": "Only operational metadata is logged (prediction result, latency). No image content or user-identifiable data.",
        "metrics": "Prometheus metrics contain only aggregate counters and histograms. No personal data.",
        "tracking": "No cookies, sessions, or user tracking.",
        "gdpr": "No personal data is collected or stored. Fully stateless service.",
        "details": "See /docs or PRIVACY.md in the repository for full privacy policy.",
    }


@router.get("/")
async def root() -> dict[str, Any]:
    """API root endpoint.

    Returns:
        Basic API information.
    """
    return {
        "name": "AI Product Photo Detector",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "healthz": "/healthz",
        "metrics": "/metrics",
        "drift": "/drift",
        "privacy": "/privacy",
    }
