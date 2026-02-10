"""Pydantic schemas for API request/response validation."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceLevel(StrEnum):
    """Confidence level categories."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionResult(StrEnum):
    """Prediction result categories."""

    REAL = "real"
    AI_GENERATED = "ai_generated"


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""

    prediction: PredictionResult = Field(description="Classification result")
    probability: float = Field(
        ge=0.0, le=1.0, description="Probability of being AI-generated (0.0 - 1.0)"
    )
    confidence: ConfidenceLevel = Field(description="Confidence level based on probability")
    inference_time_ms: float = Field(ge=0, description="Inference time in milliseconds")
    model_version: str = Field(description="Model version used for prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "ai_generated",
                "probability": 0.87,
                "confidence": "high",
                "inference_time_ms": 45.2,
                "model_version": "1.0.0",
            }
        }
    )


class HealthStatus(StrEnum):
    """Health status categories."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Response schema for /health endpoint (readiness check)."""

    status: HealthStatus = Field(description="Current health status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_version: str = Field(description="Loaded model version")
    uptime_seconds: float = Field(ge=0, description="Server uptime in seconds")
    active_requests: int = Field(default=0, description="Number of in-flight requests")
    drift_detected: bool = Field(default=False, description="Whether drift has been detected")
    predictions_total: int = Field(default=0, description="Total predictions served")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "uptime_seconds": 3600.5,
                "active_requests": 2,
                "drift_detected": False,
                "predictions_total": 1542,
            }
        }
    )


class LivenessResponse(BaseModel):
    """Response schema for /healthz (liveness probe)."""

    alive: bool = Field(description="Whether the process is alive")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Invalid image format",
                "detail": "Supported formats: JPEG, PNG, WebP",
            }
        }
    )


class BatchItemResult(BaseModel):
    """Result for a single item in batch prediction."""

    filename: str = Field(description="Original filename")
    prediction: PredictionResult | None = Field(description="Classification result")
    probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Probability of being AI-generated"
    )
    confidence: ConfidenceLevel | None = Field(default=None, description="Confidence level")
    error: str | None = Field(default=None, description="Error message if processing failed")


class BatchPredictResponse(BaseModel):
    """Response schema for /predict/batch endpoint."""

    results: list[BatchItemResult] = Field(description="Prediction results for each image")
    total: int = Field(description="Total number of images processed")
    successful: int = Field(description="Number of successful predictions")
    failed: int = Field(description="Number of failed predictions")
    total_inference_time_ms: float = Field(ge=0, description="Total inference time in milliseconds")
    model_version: str = Field(description="Model version used for predictions")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "filename": "product1.jpg",
                        "prediction": "ai_generated",
                        "probability": 0.87,
                        "confidence": "high",
                        "error": None,
                    },
                    {
                        "filename": "product2.jpg",
                        "prediction": "real",
                        "probability": 0.12,
                        "confidence": "high",
                        "error": None,
                    },
                ],
                "total": 2,
                "successful": 2,
                "failed": 0,
                "total_inference_time_ms": 89.5,
                "model_version": "1.0.0",
            }
        }
    )
