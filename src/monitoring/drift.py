"""Drift detection for model monitoring.

Detects distribution shift in incoming data that may indicate
the model needs retraining.
"""

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftMetrics:
    """Drift detection metrics."""

    timestamp: str
    window_size: int
    mean_probability: float
    std_probability: float
    low_confidence_ratio: float
    prediction_ratio: dict[str, float]
    drift_detected: bool
    drift_score: float
    alerts: list[str] = field(default_factory=list)


class DriftDetector:
    """Monitors prediction drift to detect distribution shift.

    Tracks:
    - Mean prediction probability
    - Prediction confidence distribution
    - Class prediction ratios

    Alerts when metrics deviate significantly from baseline.
    """

    def __init__(
        self,
        window_size: int = 1000,
        confidence_threshold: float = 0.3,
        drift_threshold: float = 0.15,
        baseline_path: Path | None = None,
    ) -> None:
        """Initialize drift detector.

        Args:
            window_size: Number of predictions to keep in sliding window.
            confidence_threshold: Threshold for low confidence predictions.
            drift_threshold: Threshold for triggering drift alerts.
            baseline_path: Path to baseline metrics file.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.drift_threshold = drift_threshold

        # Sliding window of predictions (thread-safe access via lock)
        self._lock = threading.Lock()
        self.predictions: deque[dict[str, Any]] = deque(maxlen=window_size)

        # Baseline metrics (from training data or initial deployment)
        self.baseline: dict[str, Any] | None = None
        if baseline_path and baseline_path.exists():
            self.load_baseline(baseline_path)

    def load_baseline(self, path: Path) -> None:
        """Load baseline metrics from file.

        Args:
            path: Path to baseline JSON file.
        """
        with open(path) as f:
            self.baseline = json.load(f)
        logger.info("Loaded drift baseline", path=str(path))

    def save_baseline(self, path: Path) -> None:
        """Save current metrics as baseline.

        Args:
            path: Path to save baseline JSON.
        """
        if len(list(self.predictions)) < self.window_size // 2:
            logger.warning("Not enough data to save baseline")
            return

        metrics = self._compute_metrics()
        baseline = {
            "mean_probability": metrics.mean_probability,
            "std_probability": metrics.std_probability,
            "low_confidence_ratio": metrics.low_confidence_ratio,
            "prediction_ratio": metrics.prediction_ratio,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)

        self.baseline = baseline
        logger.info("Saved drift baseline", path=str(path))

    def record_prediction(
        self,
        probability: float,
        prediction: str,
    ) -> None:
        """Record a prediction for drift monitoring.

        Args:
            probability: Prediction probability (0-1).
            prediction: Prediction label ("real" or "ai_generated").
        """
        with self._lock:
            self.predictions.append(
                {
                    "probability": probability,
                    "prediction": prediction,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    def _compute_metrics(self) -> DriftMetrics:
        """Compute current drift metrics.

        Returns:
            DriftMetrics with current statistics.
        """
        with self._lock:
            snapshot = list(self.predictions)

        if len(snapshot) == 0:
            return DriftMetrics(
                timestamp=datetime.now(UTC).isoformat(),
                window_size=0,
                mean_probability=0.5,
                std_probability=0.0,
                low_confidence_ratio=0.0,
                prediction_ratio={"real": 0.5, "ai_generated": 0.5},
                drift_detected=False,
                drift_score=0.0,
            )

        probs = np.array([p["probability"] for p in snapshot])
        preds = [p["prediction"] for p in snapshot]

        # Compute statistics
        mean_prob = float(np.mean(probs))
        std_prob = float(np.std(probs))

        # Low confidence: predictions near 0.5
        low_conf_mask = np.abs(probs - 0.5) < self.confidence_threshold
        low_conf_ratio = float(np.mean(low_conf_mask))

        # Prediction ratios
        ai_ratio = sum(1 for p in preds if p == "ai_generated") / len(preds)
        pred_ratio = {"real": 1 - ai_ratio, "ai_generated": ai_ratio}

        # Compute drift score
        drift_score = 0.0
        alerts = []

        if self.baseline:
            # Compare to baseline
            prob_drift = abs(mean_prob - self.baseline["mean_probability"])
            conf_drift = abs(low_conf_ratio - self.baseline["low_confidence_ratio"])

            drift_score = max(prob_drift, conf_drift)

            if prob_drift > self.drift_threshold:
                alerts.append(f"Probability drift: {prob_drift:.3f}")

            if conf_drift > self.drift_threshold:
                alerts.append(f"Confidence drift: {conf_drift:.3f}")

            if abs(ai_ratio - self.baseline["prediction_ratio"]["ai_generated"]) > 0.2:
                alerts.append(f"Prediction ratio drift: AI {ai_ratio:.1%}")

        return DriftMetrics(
            timestamp=datetime.now(UTC).isoformat(),
            window_size=len(snapshot),
            mean_probability=mean_prob,
            std_probability=std_prob,
            low_confidence_ratio=low_conf_ratio,
            prediction_ratio=pred_ratio,
            drift_detected=len(alerts) > 0,
            drift_score=drift_score,
            alerts=alerts,
        )

    def check_drift(self) -> DriftMetrics:
        """Check for drift and return metrics.

        Returns:
            Current drift metrics with alerts.
        """
        metrics = self._compute_metrics()

        if metrics.drift_detected:
            logger.warning(
                "Drift detected",
                drift_score=metrics.drift_score,
                alerts=metrics.alerts,
            )

        return metrics

    def get_status(self) -> dict[str, Any]:
        """Get current monitoring status.

        Returns:
            Status dictionary for health checks.
        """
        metrics = self._compute_metrics()

        return {
            "window_size": metrics.window_size,
            "window_capacity": self.window_size,
            "mean_probability": round(metrics.mean_probability, 4),
            "low_confidence_ratio": round(metrics.low_confidence_ratio, 4),
            "drift_detected": metrics.drift_detected,
            "drift_score": round(metrics.drift_score, 4),
            "has_baseline": self.baseline is not None,
        }
