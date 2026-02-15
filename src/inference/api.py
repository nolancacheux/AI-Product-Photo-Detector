"""FastAPI application for AI Product Photo Detector.

Production-hardened API with:
- API versioning (/v1 prefix + backward-compatible root routes)
- Security headers middleware
- GZip response compression
- Graceful shutdown with request draining
- Structured error responses with request_id
- Observability (request tracking, Prometheus metrics)
"""

import asyncio
import os
import signal
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import Response

from src.inference import state
from src.inference.explainer import GradCAMExplainer
from src.inference.predictor import Predictor
from src.inference.rate_limit import limiter
from src.inference.routes import info_router, monitoring_router, predict_router
from src.inference.routes.v1 import (
    info_router as v1_info_router,
)
from src.inference.routes.v1 import (
    monitoring_router as v1_monitoring_router,
)
from src.inference.routes.v1 import (
    predict_router as v1_predict_router,
)
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    ERRORS_TOTAL,
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

# ---------------------------------------------------------------------------
# API version constant
# ---------------------------------------------------------------------------
API_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Graceful shutdown state
# ---------------------------------------------------------------------------
_shutdown_event: asyncio.Event | None = None


def _get_shutdown_event() -> asyncio.Event:
    """Lazy-init the shutdown event (must be called inside a running loop)."""
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event


# ---------------------------------------------------------------------------
# Prediction endpoints (used by Cache-Control logic)
# ---------------------------------------------------------------------------
_PREDICTION_PATHS = frozenset(
    {
        "/predict",
        "/predict/batch",
        "/predict/explain",
        "/v1/predict",
        "/v1/predict/batch",
        "/v1/predict/explain",
    }
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler with graceful shutdown support."""
    # --- Startup ---
    setup_logging(level="INFO", json_format=True)
    logger.info("Starting AI Product Photo Detector API", api_version=API_VERSION)

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

    # Register SIGTERM handler for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = _get_shutdown_event()

    def _handle_sigterm(sig: int, frame: Any) -> None:
        logger.info(
            "Received shutdown signal",
            signal=sig,
            uptime_seconds=round(time.time() - state.start_time, 2),
            predictions_served=state.get_total_predictions(),
        )
        loop.call_soon_threadsafe(shutdown_event.set)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)
    except ValueError:
        pass  # signal only works in main thread; skip in test/worker threads

    yield

    # --- Shutdown ---
    logger.info("Initiating graceful shutdown — draining active requests")

    # Wait for in-flight requests to complete (max 30s)
    from src.monitoring.metrics import ACTIVE_REQUESTS

    drain_deadline = time.monotonic() + 30.0
    while time.monotonic() < drain_deadline:
        active = int(ACTIVE_REQUESTS._value.get())
        if active <= 0:
            break
        logger.info("Draining requests", active_requests=active)
        await asyncio.sleep(0.5)

    uptime = round(time.time() - state.start_time, 2)
    logger.info(
        "Shutdown complete",
        uptime_seconds=uptime,
        predictions_served=state.get_total_predictions(),
    )


# ---------------------------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Product Photo Detector",
    description="Detect AI-generated product photos in e-commerce listings",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware stack (order matters — outermost first)
# ---------------------------------------------------------------------------

# 1. GZip compression (outermost so it compresses all responses)
app.add_middleware(GZipMiddleware, minimum_size=500)

# 2. Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# 3. CORS
_default_origins = [
    "https://ai-product-detector-714127049161.europe-west1.run.app",
    "http://localhost:8080",
    "http://localhost:8501",
    "http://localhost:3000",
]
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
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
# Middleware: Security headers + API version
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next: Any) -> Response:
    """Inject security headers and API version on every response."""
    response: Response = await call_next(request)

    # API version header
    response.headers["X-API-Version"] = API_VERSION

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "font-src 'self' https://cdn.jsdelivr.net"
    )

    # No-cache for prediction endpoints (contain sensitive classification data)
    if request.url.path in _PREDICTION_PATHS:
        response.headers["Cache-Control"] = "no-store"

    return response


# ---------------------------------------------------------------------------
# Middleware: Observability (request tracking, metrics, request ID)
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
        ERRORS_TOTAL.labels(type="unhandled_exception", endpoint=endpoint).inc()
        track_request_end()
        raise

    duration = time.monotonic() - start
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(response.status_code),
    ).inc()

    # Classify errors for metrics
    sc = response.status_code
    if 400 <= sc < 500:
        ERRORS_TOTAL.labels(type="client_error", endpoint=endpoint).inc()
    elif sc >= 500:
        ERRORS_TOTAL.labels(type="server_error", endpoint=endpoint).inc()

    resp_size = response.headers.get("content-length")
    if resp_size:
        RESPONSE_SIZE_BYTES.observe(int(resp_size))

    response.headers["X-Request-ID"] = req_id
    track_request_end()
    return response


# ---------------------------------------------------------------------------
# Structured error handler
# ---------------------------------------------------------------------------
def _classify_error(status_code: int) -> str:
    """Classify HTTP status code into client_error or server_error."""
    if 400 <= status_code < 500:
        return "client_error"
    return "server_error"


@app.exception_handler(HTTPException)
async def structured_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured error responses with request_id and classification."""
    req_id = request.headers.get("X-Request-ID", "unknown")

    # Support both string and dict detail formats
    if isinstance(exc.detail, dict):
        error = exc.detail.get("error", "Error")
        detail = exc.detail.get("detail", str(exc.detail))
    else:
        error = "Error"
        detail = str(exc.detail)

    body = {
        "error": error,
        "detail": detail,
        "request_id": req_id,
        "status_code": exc.status_code,
        "error_class": _classify_error(exc.status_code),
    }

    return JSONResponse(status_code=exc.status_code, content=body)


# ---------------------------------------------------------------------------
# Include route modules — backward-compatible root + versioned /v1
# ---------------------------------------------------------------------------

# Root routes (backward compatibility)
app.include_router(predict_router)
app.include_router(monitoring_router)
app.include_router(info_router)

# Versioned routes under /v1
app.include_router(v1_predict_router, prefix="/v1")
app.include_router(v1_monitoring_router, prefix="/v1")
app.include_router(v1_info_router, prefix="/v1")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
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
