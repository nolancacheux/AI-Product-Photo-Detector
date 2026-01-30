"""FastAPI application for AI Product Photo Detector."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.inference.predictor import Predictor
from src.inference.schemas import (
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    PredictResponse,
)
from src.utils.config import get_settings, load_yaml_config
from src.utils.logger import get_logger, setup_logging

# Explainability (lazy loaded)
explainable_predictor = None

# Metrics
REQUEST_COUNT = Counter(
    "predictions_total",
    "Total number of predictions",
    ["status", "prediction"],
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Global state
predictor: Predictor | None = None
start_time: float = 0.0
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    global predictor, start_time

    # Startup
    setup_logging(level="INFO", json_format=True)
    logger.info("Starting AI Product Photo Detector API")

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
    else:
        logger.warning("Model not loaded - predictions will fail")

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        Health status with model info.
    """
    global predictor, start_time

    uptime = time.time() - start_time if start_time > 0 else 0.0
    model_loaded = predictor is not None and predictor.is_ready()

    return HealthResponse(
        status=HealthStatus.HEALTHY if model_loaded else HealthStatus.UNHEALTHY,
        model_loaded=model_loaded,
        model_version=predictor.model_version if predictor else "unknown",
        uptime_seconds=round(uptime, 2),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Prediction"],
)
async def predict(
    file: UploadFile = File(..., description="Image file to analyze"),
) -> PredictResponse:
    """Predict if an image is AI-generated.

    Accepts JPEG, PNG, or WebP images up to 10MB.

    Returns:
        Prediction with probability score and confidence level.
    """
    global predictor

    # Check if model is loaded
    if predictor is None or not predictor.is_ready():
        REQUEST_COUNT.labels(status="error", prediction="none").inc()
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Model not loaded"},
        )

    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        REQUEST_COUNT.labels(status="error", prediction="none").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid image format",
                "detail": f"Supported formats: JPEG, PNG, WebP. Got: {file.content_type}",
            },
        )

    # Read file
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        REQUEST_COUNT.labels(status="error", prediction="none").inc()
        raise HTTPException(
            status_code=413,
            detail={
                "error": "File too large",
                "detail": f"Maximum file size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            },
        )

    # Make prediction
    try:
        with REQUEST_LATENCY.time():
            result = predictor.predict_from_bytes(contents)

        REQUEST_COUNT.labels(status="success", prediction=result.prediction.value).inc()

        logger.info(
            "Prediction complete",
            prediction=result.prediction.value,
            probability=result.probability,
            inference_time_ms=result.inference_time_ms,
        )

        return result

    except ValueError as e:
        REQUEST_COUNT.labels(status="error", prediction="none").inc()
        raise HTTPException(
            status_code=400,
            detail={"error": "Processing error", "detail": str(e)},
        ) from e


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


@app.get("/", tags=["Info"])
async def root() -> dict:
    """API root endpoint.

    Returns:
        Basic API information.
    """
    return {
        "name": "AI Product Photo Detector",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post(
    "/explain",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Explainability"],
)
async def explain(
    file: UploadFile = File(..., description="Image file to analyze and explain"),
    alpha: float = 0.5,
) -> Response:
    """Generate GradCAM explanation for prediction.

    Returns the image with a heatmap overlay showing which regions
    the model focused on for its prediction.

    Args:
        file: Image file to analyze.
        alpha: Heatmap overlay transparency (0-1). Default 0.5.

    Returns:
        PNG image with GradCAM heatmap overlay.
    """
    global predictor, explainable_predictor

    # Check if model is loaded
    if predictor is None or not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Model not loaded"},
        )

    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid image format",
                "detail": f"Supported formats: JPEG, PNG, WebP. Got: {file.content_type}",
            },
        )

    # Read file
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "File too large",
                "detail": f"Maximum file size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            },
        )

    try:
        # Lazy load explainability module
        if explainable_predictor is None:
            from src.inference.explainability import ExplainablePredictor

            explainable_predictor = ExplainablePredictor(
                model=predictor.model,
                device=predictor.device,
                transform=predictor.transform,
            )

        # Generate explanation
        overlay_bytes, probability = explainable_predictor.explain_to_bytes(
            contents, alpha=alpha
        )

        logger.info(
            "Explanation generated",
            probability=probability,
            alpha=alpha,
        )

        return Response(
            content=overlay_bytes,
            media_type="image/png",
            headers={
                "X-Prediction-Probability": str(round(probability, 4)),
                "X-Prediction-Class": "ai_generated" if probability > 0.5 else "real",
            },
        )

    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Processing error", "detail": str(e)},
        ) from e


def main() -> None:
    """Run the API server."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
