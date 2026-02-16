"""Shared confidence level classification logic."""

from src.inference.schemas import ConfidenceLevel

# Default thresholds for confidence classification
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.3


def classify_confidence(
    probability: float,
    high_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    low_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> ConfidenceLevel:
    """Determine confidence level from prediction probability.

    Confidence is based on distance from the 0.5 decision boundary:
    - HIGH: distance > (high_threshold - 0.5)
    - LOW: distance < (0.5 - low_threshold)
    - MEDIUM: everything in between

    Args:
        probability: Prediction probability (0.0 to 1.0).
        high_threshold: Threshold above which confidence is high.
        low_threshold: Threshold below which confidence is low.

    Returns:
        Confidence level enum value.
    """
    distance = abs(probability - 0.5)

    if distance > (high_threshold - 0.5):
        return ConfidenceLevel.HIGH
    elif distance < (0.5 - low_threshold):
        return ConfidenceLevel.LOW
    return ConfidenceLevel.MEDIUM
