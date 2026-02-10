"""FastAPI application for AI Product Photo Detector."""

import os
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from src.inference.auth import verify_api_key
from src.inference.explainer import GradCAMExplainer
from src.inference.predictor import Predictor
from src.inference.schemas import (
    BatchItemResult,
    BatchPredictResponse,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    HealthStatus,
    LivenessResponse,
    PredictResponse,
)
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    ACTIVE_REQUESTS,
    BATCH_LATENCY,
    BATCH_PREDICTIONS_TOTAL,
    BATCH_SIZE_HISTOGRAM,
    ERRORS_TOTAL,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    IMAGE_VALIDATION_ERRORS,
    MODEL_LOADED,
    PREDICTIONS_TOTAL,
    REQUEST_SIZE_BYTES,
    RESPONSE_SIZE_BYTES,
    set_app_info,
    set_model_info,
    track_request_end,
    track_request_start,
)
from src.monitoring.metrics import (
    record_prediction as record_prediction_metrics,
)
from src.utils.config import get_settings, load_yaml_config
from src.utils.logger import get_logger, set_request_id, setup_logging


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP behind Cloud Run / reverse proxies.

    Cloud Run sets X-Forwarded-For with the real client IP as first entry.
    Falls back to request.client.host if header is absent.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


# Rate limiting — use real client IP behind proxies, with global default
limiter = Limiter(key_func=_get_real_client_ip, default_limits=["200/minute"])

# Global state
predictor: Predictor | None = None
explainer: GradCAMExplainer | None = None
drift_detector: DriftDetector | None = None
start_time: float = 0.0
_predictions_lock = threading.Lock()
_total_predictions: int = 0
logger = get_logger(__name__)


def _increment_predictions() -> int:
    """Atomically increment and return total predictions count."""
    global _total_predictions
    with _predictions_lock:
        _total_predictions += 1
        return _total_predictions


def _get_total_predictions() -> int:
    """Thread-safe read of total predictions."""
    with _predictions_lock:
        return _total_predictions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    global predictor, explainer, drift_detector, start_time

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
    predictor = Predictor(model_path=model_path)
    start_time = time.time()

    if predictor.is_ready():
        logger.info("Model loaded successfully")
        set_model_info(
            name="ai-product-photo-detector",
            version=predictor.model_version,
            architecture="efficientnet",
            parameters=0,
        )
        MODEL_LOADED.set(1)
    else:
        logger.warning("Model not loaded - predictions will fail")
        MODEL_LOADED.set(0)

    # Initialize Grad-CAM explainer
    explainer = GradCAMExplainer(model_path=model_path)
    if explainer.is_ready():
        logger.info("GradCAM explainer ready")
    else:
        logger.warning("GradCAM explainer not loaded - explain endpoint will fail")

    # Initialize drift detector
    drift_detector = DriftDetector(window_size=1000)

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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total for batch
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ---------------------------------------------------------------------------
# Middleware: request tracking, request ID, HTTP metrics
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next) -> Response:  # noqa: ANN001
    """Track HTTP metrics, request size, response size, and inject request ID."""
    # Inject request ID from header or generate
    incoming_id = request.headers.get("X-Request-ID")
    req_id = set_request_id(incoming_id)

    # Track Cloud Trace context for correlated logging
    trace_header = request.headers.get("X-Cloud-Trace-Context")
    if trace_header:
        structlog.contextvars.bind_contextvars(trace_id=trace_header.split("/")[0])

    # Request size (content-length if present)
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

    # Response size
    resp_size = response.headers.get("content-length")
    if resp_size:
        RESPONSE_SIZE_BYTES.observe(int(resp_size))

    response.headers["X-Request-ID"] = req_id
    track_request_end()
    return response


# ---------------------------------------------------------------------------
# Health endpoints: liveness + readiness
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=LivenessResponse, tags=["Health"])
async def liveness() -> LivenessResponse:
    """Kubernetes / Cloud Run liveness probe.

    Returns 200 as long as the process is alive.
    Does NOT check model state — that's the readiness probe's job.
    """
    return LivenessResponse(alive=True)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Readiness probe — checks model and drift status.

    Returns:
        Health status with model info, active requests, drift status.
    """
    global predictor, drift_detector, start_time

    uptime = time.time() - start_time if start_time > 0 else 0.0
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
        predictions_total=_get_total_predictions(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse},
    },
    tags=["Prediction"],
)
@limiter.limit("30/minute")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> PredictResponse:
    """Predict if an image is AI-generated.

    Accepts JPEG, PNG, or WebP images up to 5MB.
    Requires authentication when API_KEYS or REQUIRE_AUTH is configured.

    Returns:
        Prediction with probability score and confidence level.
    """
    global predictor, drift_detector

    # Check if model is loaded
    if predictor is None or not predictor.is_ready():
        PREDICTIONS_TOTAL.labels(
            status="error",
            prediction="none",
            confidence="none",
        ).inc()
        ERRORS_TOTAL.labels(type="model_not_loaded", endpoint="/predict").inc()
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Model not loaded"},
        )

    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        PREDICTIONS_TOTAL.labels(
            status="error",
            prediction="none",
            confidence="none",
        ).inc()
        IMAGE_VALIDATION_ERRORS.labels(error_type="invalid_format").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid image format",
                "detail": f"Supported formats: JPEG, PNG, WebP. Got: {file.content_type}",
            },
        )

    # Read file with size limit — avoid loading huge payloads into memory
    contents = await file.read(MAX_FILE_SIZE + 1)

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        PREDICTIONS_TOTAL.labels(
            status="error",
            prediction="none",
            confidence="none",
        ).inc()
        IMAGE_VALIDATION_ERRORS.labels(error_type="file_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail={
                "error": "File too large",
                "detail": f"Maximum file size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            },
        )

    # Make prediction
    try:
        pred_start = time.monotonic()
        result = predictor.predict_from_bytes(contents)
        latency = time.monotonic() - pred_start

        record_prediction_metrics(
            prediction=result.prediction.value,
            probability=result.probability,
            confidence=result.confidence.value,
            latency_seconds=latency,
            success=True,
        )
        _increment_predictions()

        if drift_detector is not None:
            drift_detector.record_prediction(
                result.probability,
                result.prediction.value,
            )

        logger.info(
            "Prediction complete",
            prediction=result.prediction.value,
            probability=result.probability,
            inference_time_ms=result.inference_time_ms,
        )

        return result

    except ValueError as e:
        PREDICTIONS_TOTAL.labels(
            status="error",
            prediction="none",
            confidence="none",
        ).inc()
        ERRORS_TOTAL.labels(type="processing_error", endpoint="/predict").inc()
        raise HTTPException(
            status_code=400,
            detail={"error": "Processing error", "detail": str(e)},
        ) from e
    except Exception as exc:
        PREDICTIONS_TOTAL.labels(
            status="error",
            prediction="none",
            confidence="none",
        ).inc()
        ERRORS_TOTAL.labels(type="internal_error", endpoint="/predict").inc()
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal error", "detail": "An unexpected error occurred"},
        ) from exc


MAX_BATCH_SIZE = 10


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse},
    },
    tags=["Prediction"],
)
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files to analyze (max 10)"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> BatchPredictResponse:
    """Predict if multiple images are AI-generated.

    Accepts up to 10 JPEG, PNG, or WebP images.
    Each image must be under 5MB. Total payload under 50MB.
    Requires authentication when API_KEYS or REQUIRE_AUTH is configured.

    Returns:
        Batch prediction results with individual results for each image.
    """
    global predictor

    # Check if model is loaded
    if predictor is None or not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Model not loaded"},
        )

    # Check batch size
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Batch too large",
                "detail": f"Maximum batch size: {MAX_BATCH_SIZE}. Got: {len(files)}",
            },
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "Empty batch", "detail": "No files provided"},
        )

    BATCH_SIZE_HISTOGRAM.observe(len(files))
    results: list[BatchItemResult] = []
    successful = 0
    failed = 0
    total_bytes = 0
    batch_start = time.monotonic()

    for file in files:
        # Validate content type
        if file.content_type not in ALLOWED_TYPES:
            results.append(
                BatchItemResult(
                    filename=file.filename or "unknown",
                    prediction=None,
                    probability=None,
                    confidence=None,
                    error=f"Invalid format: {file.content_type}. Supported: JPEG, PNG, WebP",
                )
            )
            failed += 1
            PREDICTIONS_TOTAL.labels(
                status="error",
                prediction="none",
                confidence="none",
            ).inc()
            continue

        # Read file with bounded size to avoid memory exhaustion
        contents = await file.read(MAX_FILE_SIZE + 1)
        total_bytes += len(contents)

        # Check total batch payload size
        if total_bytes > MAX_BATCH_TOTAL_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "Batch payload too large",
                    "detail": f"Total batch size exceeds {MAX_BATCH_TOTAL_SIZE // (1024 * 1024)}MB limit",
                },
            )

        # Check individual file size
        if len(contents) > MAX_FILE_SIZE:
            results.append(
                BatchItemResult(
                    filename=file.filename or "unknown",
                    prediction=None,
                    probability=None,
                    confidence=None,
                    error=f"File too large: {len(contents) / (1024 * 1024):.1f}MB. Max: {MAX_FILE_SIZE // (1024 * 1024)}MB",
                )
            )
            failed += 1
            PREDICTIONS_TOTAL.labels(
                status="error",
                prediction="none",
                confidence="none",
            ).inc()
            continue

        # Make prediction
        try:
            result = predictor.predict_from_bytes(contents)
            results.append(
                BatchItemResult(
                    filename=file.filename or "unknown",
                    prediction=result.prediction,
                    probability=result.probability,
                    confidence=result.confidence,
                    error=None,
                )
            )
            successful += 1
            _increment_predictions()
            PREDICTIONS_TOTAL.labels(
                status="success",
                prediction=result.prediction.value,
                confidence=result.confidence.value,
            ).inc()

        except Exception as e:
            results.append(
                BatchItemResult(
                    filename=file.filename or "unknown",
                    prediction=None,
                    probability=None,
                    confidence=None,
                    error=str(e),
                )
            )
            failed += 1
            PREDICTIONS_TOTAL.labels(
                status="error",
                prediction="none",
                confidence="none",
            ).inc()

    total_time_ms = (time.monotonic() - batch_start) * 1000
    BATCH_LATENCY.observe(total_time_ms / 1000)
    BATCH_PREDICTIONS_TOTAL.labels(status="success" if failed == 0 else "partial").inc()

    logger.info(
        "Batch prediction complete",
        total=len(files),
        successful=successful,
        failed=failed,
        total_time_ms=total_time_ms,
    )

    return BatchPredictResponse(
        results=results,
        total=len(files),
        successful=successful,
        failed=failed,
        total_inference_time_ms=round(total_time_ms, 2),
        model_version=predictor.model_version,
    )


@app.post(
    "/predict/explain",
    response_model=ExplainResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse},
    },
    tags=["Prediction"],
)
@limiter.limit("10/minute")
async def predict_explain(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze with Grad-CAM"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> ExplainResponse:
    """Generate prediction with Grad-CAM heatmap explanation.

    Accepts JPEG, PNG, or WebP images up to 5MB.
    Returns the prediction plus a base64-encoded JPEG heatmap overlay
    showing which regions of the image influenced the model's decision.
    """
    global explainer

    if explainer is None or not explainer.is_ready():
        ERRORS_TOTAL.labels(type="explainer_not_loaded", endpoint="/predict/explain").inc()
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Explainer not loaded"},
        )

    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        IMAGE_VALIDATION_ERRORS.labels(error_type="invalid_format").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid image format",
                "detail": f"Supported formats: JPEG, PNG, WebP. Got: {file.content_type}",
            },
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        IMAGE_VALIDATION_ERRORS.labels(error_type="file_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail={
                "error": "File too large",
                "detail": f"Maximum file size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            },
        )

    try:
        result = explainer.explain(contents)
        _increment_predictions()

        logger.info(
            "Explain prediction complete",
            prediction=result["prediction"],
            probability=result["probability"],
            inference_time_ms=result["inference_time_ms"],
        )

        return ExplainResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            confidence=result["confidence"],
            heatmap_base64=result["heatmap_base64"],
            inference_time_ms=result["inference_time_ms"],
            model_version=explainer.model_version,
        )

    except ValueError as e:
        ERRORS_TOTAL.labels(type="processing_error", endpoint="/predict/explain").inc()
        raise HTTPException(
            status_code=400,
            detail={"error": "Processing error", "detail": str(e)},
        ) from e
    except Exception as exc:
        ERRORS_TOTAL.labels(type="internal_error", endpoint="/predict/explain").inc()
        logger.exception("Unexpected error during explain prediction")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal error", "detail": "An unexpected error occurred"},
        ) from exc


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """Expose Prometheus metrics.

    Returns:
        Prometheus-formatted metrics.
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


@app.get("/drift", tags=["Monitoring"])
async def drift_status() -> dict[str, Any]:
    """Get drift detection status with full metrics.

    Returns:
        Current drift monitoring status including alerts.
    """
    global drift_detector

    if drift_detector is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Drift detector not initialized"},
        )

    drift_metrics = drift_detector.check_drift()
    return asdict(drift_metrics)


@app.get("/privacy", tags=["Info"])
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


@app.get("/", tags=["Info"])
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
