"""V1 prediction routes — thin wrapper over shared implementation.

All prediction logic lives in src.inference.routes.predict.
This module creates a separate APIRouter so FastAPI can mount it
under the /v1 prefix without duplicating endpoint functions.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from src.inference.auth import verify_api_key
from src.inference.rate_limit import limiter
from src.inference.routes.predict import (
    predict as _predict,
)
from src.inference.routes.predict import (
    predict_batch as _predict_batch,
)
from src.inference.routes.predict import (
    predict_explain as _predict_explain,
)
from src.inference.schemas import (
    BatchPredictResponse,
    ErrorResponse,
    ExplainResponse,
    PredictResponse,
)

router = APIRouter(tags=["Prediction"])

_error_responses = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse},
}


@router.post("/predict", response_model=PredictResponse, responses=_error_responses)
@limiter.limit("30/minute")
async def predict_v1(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> PredictResponse:
    """Predict if an image is AI-generated (v1)."""
    return await _predict(request=request, file=file)


@router.post("/predict/batch", response_model=BatchPredictResponse, responses=_error_responses)
@limiter.limit("5/minute")
async def predict_batch_v1(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files to analyze (max 10)"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> BatchPredictResponse:
    """Batch predict if images are AI-generated (v1)."""
    return await _predict_batch(request=request, files=files)


@router.post("/predict/explain", response_model=ExplainResponse, responses=_error_responses)
@limiter.limit("10/minute")
async def predict_explain_v1(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze with Grad-CAM"),
    _: Annotated[bool, Depends(verify_api_key)] = True,
) -> ExplainResponse:
    """Generate prediction with Grad-CAM heatmap explanation (v1)."""
    return await _predict_explain(request=request, file=file)
