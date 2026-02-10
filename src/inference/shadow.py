"""Shadow/comparison mode for A/B model testing."""

import os

from src.inference.predictor import Predictor
from src.inference.schemas import CompareResponse, PredictResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

_shadow_predictor: Predictor | None = None
_shadow_load_attempted: bool = False


def get_shadow_predictor() -> Predictor | None:
    """Lazily load the shadow model predictor.

    Returns:
        Shadow Predictor instance if SHADOW_MODEL_PATH is set and valid, else None.
    """
    global _shadow_predictor, _shadow_load_attempted

    if _shadow_load_attempted:
        return _shadow_predictor

    _shadow_load_attempted = True
    shadow_path = os.getenv("SHADOW_MODEL_PATH")

    if not shadow_path:
        logger.info("No SHADOW_MODEL_PATH configured, shadow mode disabled")
        return None

    try:
        _shadow_predictor = Predictor(model_path=shadow_path)
        if _shadow_predictor.is_ready():
            logger.info("Shadow model loaded", path=shadow_path)
        else:
            logger.warning("Shadow model failed to load", path=shadow_path)
            _shadow_predictor = None
    except Exception:
        logger.exception("Failed to initialize shadow predictor")
        _shadow_predictor = None

    return _shadow_predictor


def compare_predictions(
    primary_result: PredictResponse,
    shadow_result: PredictResponse | None,
) -> CompareResponse:
    """Build a CompareResponse from primary and optional shadow results.

    Args:
        primary_result: Prediction from the primary model.
        shadow_result: Prediction from the shadow model (may be None).

    Returns:
        CompareResponse with agreement and difference metrics.
    """
    if shadow_result is None:
        return CompareResponse(
            primary=primary_result,
            shadow=None,
            agreement=True,
            difference=0.0,
            message="No shadow model configured",
        )

    agreement = primary_result.prediction == shadow_result.prediction
    difference = round(abs(primary_result.probability - shadow_result.probability), 4)

    logger.info(
        "Shadow comparison",
        primary_prediction=primary_result.prediction.value,
        shadow_prediction=shadow_result.prediction.value,
        primary_prob=primary_result.probability,
        shadow_prob=shadow_result.probability,
        agreement=agreement,
        difference=difference,
    )

    return CompareResponse(
        primary=primary_result,
        shadow=shadow_result,
        agreement=agreement,
        difference=difference,
    )
