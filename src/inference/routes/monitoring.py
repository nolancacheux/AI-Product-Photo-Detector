"""Monitoring route handlers: health probes, metrics, drift.

Provides Kubernetes-compatible health probes:
- /healthz  -> liveness  (process alive)
- /readyz   -> readiness (model loaded + dependencies ready)
- /health   -> detailed  (model version, uptime, drift, memory, predictions)
- /startup  -> startup   (model loading complete)
- /metrics  -> Prometheus
- /drift    -> drift detection status
"""

import os
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from prometheus_client import generate_latest
from starlette.responses import Response

from src.inference import state
from src.inference.schemas import (
    DetailedHealthResponse,
    HealthResponse,
    HealthStatus,
    LivenessResponse,
)
from src.monitoring.metrics import ACTIVE_REQUESTS

router = APIRouter()


def _get_memory_usage_mb() -> float:
    """Read RSS memory usage from /proc/self/status (Linux) or psutil fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (OSError, ValueError):
        pass

    # Fallback: os.getpid() + resource module
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # kB -> MB on Linux
    except Exception:
        return 0.0


@router.get("/healthz", response_model=LivenessResponse, tags=["Health"])
async def liveness() -> LivenessResponse:
    """Kubernetes / Cloud Run liveness probe.

    Returns 200 as long as the process is alive.
    Does NOT check model state — that's the readiness probe's job.
    """
    return LivenessResponse(alive=True)


@router.get("/readyz", response_model=HealthResponse, tags=["Health"])
async def readiness() -> HealthResponse:
    """Readiness probe — checks model and critical dependencies.

    Returns 200 only when the model is loaded and able to serve predictions.
    Returns 503 if the model is not ready.
    """
    predictor = state.predictor
    model_loaded = predictor is not None and predictor.is_ready()

    status_code = HealthStatus.HEALTHY if model_loaded else HealthStatus.UNHEALTHY

    response = HealthResponse(
        status=status_code,
        model_loaded=model_loaded,
        model_version=predictor.model_version if predictor else "unknown",
    )

    if not model_loaded:
        raise HTTPException(status_code=503, detail=response.model_dump())

    return response


@router.get("/health", response_model=DetailedHealthResponse, tags=["Health"])
async def detailed_health() -> DetailedHealthResponse:
    """Detailed health check with full operational metrics.

    Includes model version, uptime, drift status, memory, prediction counts,
    and active request count. Useful for dashboards and deep health inspection.
    """
    predictor = state.predictor
    drift_detector = state.drift_detector

    uptime = time.time() - state.start_time if state.start_time > 0 else 0.0
    model_loaded = predictor is not None and predictor.is_ready()

    drift_detected = False
    drift_status_detail: dict[str, Any] = {}
    if drift_detector:
        drift_info = drift_detector.get_status()
        drift_detected = drift_info.get("drift_detected", False)
        drift_status_detail = drift_info

    return DetailedHealthResponse(
        status=HealthStatus.HEALTHY if model_loaded else HealthStatus.UNHEALTHY,
        model_loaded=model_loaded,
        model_version=predictor.model_version if predictor else "unknown",
        uptime_seconds=round(uptime, 2),
        active_requests=int(ACTIVE_REQUESTS._value.get()),
        drift_detected=drift_detected,
        predictions_total=state.get_total_predictions(),
        memory_usage_mb=round(_get_memory_usage_mb(), 2),
        drift_status=drift_status_detail,
        pid=os.getpid(),
    )


# Backward compatibility: /health used to be the readiness-style endpoint.
# Now it returns detailed info. The old schema fields are a superset so
# existing consumers parsing HealthResponse fields will still work.


@router.get("/startup", response_model=LivenessResponse, tags=["Health"])
async def startup_probe() -> LivenessResponse:
    """Startup probe — returns 200 once model loading is complete.

    Returns 503 while the model is still loading.
    Use as a Kubernetes startupProbe to avoid premature liveness checks.
    """
    predictor = state.predictor
    model_loaded = predictor is not None and predictor.is_ready()

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail={"alive": False, "reason": "Model still loading"},
        )

    return LivenessResponse(alive=True)


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
