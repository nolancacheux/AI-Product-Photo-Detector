"""Monitoring route handlers: /health, /healthz, /metrics, /drift."""

import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from prometheus_client import generate_latest
from starlette.responses import Response

from src.inference import state
from src.inference.schemas import (
    HealthResponse,
    HealthStatus,
    LivenessResponse,
)
from src.monitoring.metrics import ACTIVE_REQUESTS

router = APIRouter()


@router.get("/healthz", response_model=LivenessResponse, tags=["Health"])
async def liveness() -> LivenessResponse:
    """Kubernetes / Cloud Run liveness probe.

    Returns 200 as long as the process is alive.
    Does NOT check model state — that's the readiness probe's job.
    """
    return LivenessResponse(alive=True)


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Readiness probe — checks model and drift status.

    Returns:
        Health status with model info, active requests, drift status.
    """
    predictor = state.predictor
    drift_detector = state.drift_detector

    uptime = time.time() - state.start_time if state.start_time > 0 else 0.0
    model_loaded = predictor is not None and predictor.is_ready()

    drift_detected = False
    if drift_detector:
        drift_status = drift_detector.get_status()
        drift_detected = drift_status.get("drift_detected", False)

    return HealthResponse(
        status=HealthStatus.HEALTHY if model_loaded else HealthStatus.UNHEALTHY,
        model_loaded=model_loaded,
        model_version=predictor.model_version if predictor else "unknown",
        uptime_seconds=round(uptime, 2),
        active_requests=int(ACTIVE_REQUESTS._value.get()),
        drift_detected=drift_detected,
        predictions_total=state.get_total_predictions(),
    )


@router.get("/metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """Expose Prometheus metrics.

    Returns:
        Prometheus-formatted metrics.
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


@router.get("/drift", tags=["Monitoring"])
async def drift_status() -> dict[str, Any]:
    """Get drift detection status with full metrics.

    Returns:
        Current drift monitoring status including alerts.
    """
    drift_detector = state.drift_detector

    if drift_detector is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Drift detector not initialized"},
        )

    drift_metrics = drift_detector.check_drift()
    return asdict(drift_metrics)
