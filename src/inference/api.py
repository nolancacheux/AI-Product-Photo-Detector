"""FastAPI application for AI Product Photo Detector."""

import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from src.inference import state
from src.inference.explainer import GradCAMExplainer
from src.inference.predictor import Predictor
from src.inference.rate_limit import limiter
from src.inference.routes import info_router, monitoring_router, predict_router
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    MODEL_LOADED,
    REQUEST_SIZE_BYTES,
    RESPONSE_SIZE_BYTES,
    set_app_info,
    set_model_info,
    track_request_end,
    track_request_start,
)
from src.utils.config import get_settings, load_yaml_config
from src.utils.logger import get_logger, set_request_id, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    setup_logging(level="INFO", json_format=True)
    logger.info("Starting AI Product Photo Detector API")

    set_app_info("1.0.0")

    # Load config
    settings = get_settings()
    config_path = Path("configs/inference_config.yaml")

    if config_path.exists():
        config = load_yaml_config(config_path)
        model_path = config.get("model", {}).get("path", settings.model_path)
    else:
        model_path = settings.model_path

    # Initialize predictor
    state.predictor = Predictor(model_path=model_path)
    state.start_time = time.time()

    if state.predictor.is_ready():
        logger.info("Model loaded successfully")
        set_model_info(
            name="ai-product-photo-detector",
            version=state.predictor.model_version,
            architecture="efficientnet",
            parameters=0,
        )
        MODEL_LOADED.set(1)
    else:
        logger.warning("Model not loaded - predictions will fail")
        MODEL_LOADED.set(0)

    # Initialize Grad-CAM explainer
    state.explainer = GradCAMExplainer(model_path=model_path)
    if state.explainer.is_ready():
        logger.info("GradCAM explainer ready")
    else:
        logger.warning("GradCAM explainer not loaded - explain endpoint will fail")

    # Initialize drift detector
    state.drift_detector = DriftDetector(window_size=1000)

    yield

    # Shutdown
    logger.info("Shutting down API")


# Create app
app = FastAPI(
    title="AI Product Photo Detector",
    description="Detect AI-generated product photos in e-commerce listings",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# CORS middleware — restricted origins from env var
_default_origins = [
    "https://ai-product-detector-714127049161.europe-west1.run.app",
    "http://localhost:8080",
    "http://localhost:8501",
    "http://localhost:3000",
]
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
# Support both comma and pipe as separator (pipe avoids gcloud --set-env-vars escaping issues)
_separator = "|" if "|" in _origins_env else ","
_allowed_origins = [
    o.strip() for o in _origins_env.split(_separator) if o.strip()
] or _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ---------------------------------------------------------------------------
# Middleware: request tracking, request ID, HTTP metrics
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next: Any) -> Response:
    """Track HTTP metrics, request size, response size, and inject request ID."""
    incoming_id = request.headers.get("X-Request-ID")
    req_id = set_request_id(incoming_id)

    trace_header = request.headers.get("X-Cloud-Trace-Context")
    if trace_header:
        structlog.contextvars.bind_contextvars(trace_id=trace_header.split("/")[0])

    content_length = request.headers.get("content-length")
    if content_length:
        REQUEST_SIZE_BYTES.observe(int(content_length))

    track_request_start()
    endpoint = request.url.path
    method = request.method
    start = time.monotonic()

    try:
        response = await call_next(request)
    except Exception:
        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code="500",
        ).inc()
        track_request_end()
        raise

    duration = time.monotonic() - start
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(response.status_code),
    ).inc()

    resp_size = response.headers.get("content-length")
    if resp_size:
        RESPONSE_SIZE_BYTES.observe(int(resp_size))

    response.headers["X-Request-ID"] = req_id
    track_request_end()
    return response


# ---------------------------------------------------------------------------
# Include route modules
# ---------------------------------------------------------------------------
app.include_router(predict_router)
app.include_router(monitoring_router)
app.include_router(info_router)


def main() -> None:
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
