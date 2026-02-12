"""Prediction route handlers: /predict, /predict/batch, /predict/explain."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.inference import state
from src.inference.auth import verify_api_key
from src.inference.rate_limit import limiter
from src.inference.schemas import (
    BatchItemResult,
    BatchPredictResponse,
    ErrorResponse,
    ExplainResponse,
    PredictResponse,
)
from src.monitoring.metrics import (
    BATCH_LATENCY,
    BATCH_PREDICTIONS_TOTAL,
    BATCH_SIZE_HISTOGRAM,
    ERRORS_TOTAL,
    IMAGE_VALIDATION_ERRORS,
    PREDICTIONS_TOTAL,
)
from src.monitoring.metrics import (
    record_prediction as record_prediction_metrics,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total for batch
MAX_BATCH_SIZE = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse},
    },
)
@limiter.limit("30/minute")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> PredictResponse:
    """Predict if an image is AI-generated.

    Accepts JPEG, PNG, or WebP images up to 10MB.
    Requires authentication when API_KEYS or REQUIRE_AUTH is configured.

    Returns:
        Prediction with probability score and confidence level.
    """
    predictor = state.predictor
    drift_detector = state.drift_detector

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

    contents = await file.read(MAX_FILE_SIZE + 1)

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
        state.increment_predictions()

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


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse},
    },
)
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files to analyze (max 10)"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> BatchPredictResponse:
    """Predict if multiple images are AI-generated.

    Accepts up to 10 JPEG, PNG, or WebP images.
    Each image must be under 10MB. Total payload under 50MB.
    Requires authentication when API_KEYS or REQUIRE_AUTH is configured.

    Returns:
        Batch prediction results with individual results for each image.
    """
    predictor = state.predictor

    if predictor is None or not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Model not loaded"},
        )

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

        contents = await file.read(MAX_FILE_SIZE + 1)
        total_bytes += len(contents)

        if total_bytes > MAX_BATCH_TOTAL_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "Batch payload too large",
                    "detail": f"Total batch size exceeds {MAX_BATCH_TOTAL_SIZE // (1024 * 1024)}MB limit",
                },
            )

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
            state.increment_predictions()
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


@router.post(
    "/predict/explain",
    response_model=ExplainResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse},
    },
)
@limiter.limit("10/minute")
async def predict_explain(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze with Grad-CAM"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> ExplainResponse:
    """Generate prediction with Grad-CAM heatmap explanation.

    Accepts JPEG, PNG, or WebP images up to 10MB.
    Returns the prediction plus a base64-encoded JPEG heatmap overlay
    showing which regions of the image influenced the model's decision.
    """
    explainer = state.explainer

    if explainer is None or not explainer.is_ready():
        ERRORS_TOTAL.labels(type="explainer_not_loaded", endpoint="/predict/explain").inc()
        raise HTTPException(
            status_code=503,
            detail={"error": "Service unavailable", "detail": "Explainer not loaded"},
        )

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
        state.increment_predictions()

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
