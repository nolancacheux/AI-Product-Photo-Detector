"""Pydantic schemas for API request/response validation."""

from enum import Enum

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionResult(str, Enum):
    """Prediction result categories."""
    
    REAL = "real"
    AI_GENERATED = "ai_generated"


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""
    
    prediction: PredictionResult = Field(
        description="Classification result"
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of being AI-generated (0.0 - 1.0)"
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence level based on probability"
    )
    inference_time_ms: float = Field(
        ge=0,
        description="Inference time in milliseconds"
    )
    model_version: str = Field(
        description="Model version used for prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "ai_generated",
                "probability": 0.87,
                "confidence": "high",
                "inference_time_ms": 45.2,
                "model_version": "1.0.0"
            }
        }


class HealthStatus(str, Enum):
    """Health status categories."""
    
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    
    status: HealthStatus = Field(
        description="Current health status"
    )
    model_loaded: bool = Field(
        description="Whether the model is loaded"
    )
    model_version: str = Field(
        description="Loaded model version"
    )
    uptime_seconds: float = Field(
        ge=0,
        description="Server uptime in seconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    
    error: str = Field(
        description="Error type"
    )
    detail: str = Field(
        description="Detailed error message"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image format",
                "detail": "Supported formats: JPEG, PNG, WebP"
            }
        }
