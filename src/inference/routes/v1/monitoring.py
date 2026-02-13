"""V1 monitoring routes — thin wrapper over shared implementation.

All monitoring logic lives in src.inference.routes.monitoring.
This module creates a separate APIRouter for the /v1 prefix.
"""

from typing import Any

from fastapi import APIRouter
from starlette.responses import Response

from src.inference.routes.monitoring import (
    detailed_health as _detailed_health,
)
from src.inference.routes.monitoring import (
    drift_status as _drift_status,
)
from src.inference.routes.monitoring import (
    liveness as _liveness,
)
from src.inference.routes.monitoring import (
    metrics as _metrics,
)
from src.inference.routes.monitoring import (
    readiness as _readiness,
)
from src.inference.routes.monitoring import (
    startup_probe as _startup_probe,
)
from src.inference.schemas import DetailedHealthResponse, HealthResponse, LivenessResponse

router = APIRouter()


@router.get("/healthz", response_model=LivenessResponse, tags=["Health"])
async def liveness_v1() -> LivenessResponse:
    """Liveness probe (v1)."""
    return await _liveness()


@router.get("/readyz", response_model=HealthResponse, tags=["Health"])
async def readiness_v1() -> HealthResponse:
    """Readiness probe (v1)."""
    return await _readiness()


@router.get("/health", response_model=DetailedHealthResponse, tags=["Health"])
async def detailed_health_v1() -> DetailedHealthResponse:
    """Detailed health check (v1)."""
    return await _detailed_health()


@router.get("/startup", response_model=LivenessResponse, tags=["Health"])
async def startup_probe_v1() -> LivenessResponse:
    """Startup probe (v1)."""
    return await _startup_probe()


@router.get("/metrics", tags=["Monitoring"])
async def metrics_v1() -> Response:
    """Prometheus metrics (v1)."""
    return await _metrics()


@router.get("/drift", tags=["Monitoring"])
async def drift_status_v1() -> dict[str, Any]:
    """Drift detection status (v1)."""
    return await _drift_status()
