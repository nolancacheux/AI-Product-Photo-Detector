"""Shared application state for the inference API.

Central place for mutable singletons (predictor, explainer, drift detector,
counters) so route modules can import them without circular dependencies.
"""

import threading

from src.inference.explainer import GradCAMExplainer
from src.inference.predictor import Predictor
from src.monitoring.drift import DriftDetector

# Global singletons — set during lifespan startup
predictor: Predictor | None = None
explainer: GradCAMExplainer | None = None
drift_detector: DriftDetector | None = None
start_time: float = 0.0

# Thread-safe prediction counter
_predictions_lock = threading.Lock()
_total_predictions: int = 0


def increment_predictions() -> int:
    """Atomically increment and return total predictions count."""
    global _total_predictions
    with _predictions_lock:
        _total_predictions += 1
        return _total_predictions


def get_total_predictions() -> int:
    """Thread-safe read of total predictions."""
    with _predictions_lock:
        return _total_predictions
