"""Extended tests for drift detection module — edge cases and baseline logic."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.drift import DriftDetector, DriftMetrics


class TestDriftDetectorEdgeCases:
    """Edge case tests for DriftDetector."""

    def test_single_prediction(self) -> None:
        """Metrics should work with a single prediction."""
        detector = DriftDetector(window_size=100)
        detector.record_prediction(0.75, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.window_size == 1
        assert metrics.mean_probability == 0.75
        assert metrics.prediction_ratio["ai_generated"] == 1.0
        assert metrics.prediction_ratio["real"] == 0.0

    def test_all_same_prediction(self) -> None:
        """All same predictions should have 0 std."""
        detector = DriftDetector(window_size=50)
        for _ in range(50):
            detector.record_prediction(0.8, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.std_probability == 0.0
        assert metrics.prediction_ratio["ai_generated"] == 1.0
        assert metrics.prediction_ratio["real"] == 0.0

    def test_all_real_predictions(self) -> None:
        """All real predictions should have ratio of 1.0 real."""
        detector = DriftDetector(window_size=50)
        for _ in range(50):
            detector.record_prediction(0.1, "real")

        metrics = detector.check_drift()
        assert metrics.prediction_ratio["real"] == 1.0
        assert metrics.prediction_ratio["ai_generated"] == 0.0

    def test_extreme_probabilities(self) -> None:
        """Probabilities at 0.0 and 1.0 should work."""
        detector = DriftDetector(window_size=10)
        detector.record_prediction(0.0, "real")
        detector.record_prediction(1.0, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.mean_probability == 0.5
        assert metrics.std_probability == 0.5

    def test_window_sliding_evicts_old(self) -> None:
        """Sliding window should evict oldest predictions."""
        detector = DriftDetector(window_size=3)

        detector.record_prediction(0.1, "real")
        detector.record_prediction(0.2, "real")
        detector.record_prediction(0.3, "real")
        # This should evict the 0.1
        detector.record_prediction(0.9, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.window_size == 3
        # Mean should be (0.2 + 0.3 + 0.9) / 3 ≈ 0.467
        assert abs(metrics.mean_probability - (0.2 + 0.3 + 0.9) / 3) < 0.01

    def test_low_confidence_ratio_mixed(self) -> None:
        """Mixed confidence predictions should give correct ratio."""
        detector = DriftDetector(window_size=10, confidence_threshold=0.3)

        # 5 low confidence (near 0.5)
        for _ in range(5):
            detector.record_prediction(0.5, "ai_generated")
        # 5 high confidence (far from 0.5)
        for _ in range(5):
            detector.record_prediction(0.95, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.low_confidence_ratio == 0.5

    def test_zero_confidence_threshold(self) -> None:
        """With confidence_threshold=0, only exact 0.5 should count as low."""
        detector = DriftDetector(window_size=10, confidence_threshold=0.0)

        detector.record_prediction(0.5, "ai_generated")
        detector.record_prediction(0.51, "ai_generated")

        metrics = detector.check_drift()
        # With threshold=0, abs(p - 0.5) < 0 is never true
        assert metrics.low_confidence_ratio == 0.0


class TestDriftDetectorBaseline:
    """Tests for baseline loading, saving, and drift comparison."""

    def test_load_baseline(self, tmp_path) -> None:
        """Detector should load baseline from file."""
        baseline = {
            "mean_probability": 0.5,
            "std_probability": 0.1,
            "low_confidence_ratio": 0.2,
            "prediction_ratio": {"real": 0.5, "ai_generated": 0.5},
            "timestamp": "2025-01-01T00:00:00",
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        detector = DriftDetector(baseline_path=baseline_path)
        assert detector.baseline is not None
        assert detector.baseline["mean_probability"] == 0.5

    def test_load_baseline_nonexistent(self, tmp_path) -> None:
        """Non-existent baseline path should leave baseline as None."""
        detector = DriftDetector(baseline_path=tmp_path / "nonexistent.json")
        assert detector.baseline is None

    def test_save_baseline(self, tmp_path) -> None:
        """save_baseline should write metrics to file."""
        detector = DriftDetector(window_size=10)
        for _ in range(10):
            detector.record_prediction(0.7, "ai_generated")

        save_path = tmp_path / "saved_baseline.json"
        detector.save_baseline(save_path)

        assert save_path.exists()
        saved = json.loads(save_path.read_text())
        assert "mean_probability" in saved
        assert "timestamp" in saved

    def test_save_baseline_creates_dirs(self, tmp_path) -> None:
        """save_baseline should create parent directories."""
        detector = DriftDetector(window_size=10)
        for _ in range(10):
            detector.record_prediction(0.6, "real")

        save_path = tmp_path / "deep" / "nested" / "baseline.json"
        detector.save_baseline(save_path)
        assert save_path.exists()

    def test_save_baseline_insufficient_data(self, tmp_path) -> None:
        """save_baseline should warn with insufficient data."""
        detector = DriftDetector(window_size=100)
        # Only 10 predictions, need at least 50 (window_size // 2)
        for _ in range(10):
            detector.record_prediction(0.5, "real")

        save_path = tmp_path / "baseline.json"
        detector.save_baseline(save_path)
        # File should NOT be created
        assert not save_path.exists()

    def test_drift_detected_with_baseline(self, tmp_path) -> None:
        """Drift should be detected when metrics deviate from baseline."""
        baseline = {
            "mean_probability": 0.5,
            "std_probability": 0.1,
            "low_confidence_ratio": 0.1,
            "prediction_ratio": {"real": 0.5, "ai_generated": 0.5},
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        detector = DriftDetector(
            window_size=50,
            drift_threshold=0.15,
            baseline_path=baseline_path,
        )

        # Add predictions that deviate significantly from baseline
        for _ in range(50):
            detector.record_prediction(0.95, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.drift_detected is True
        assert metrics.drift_score > 0
        assert len(metrics.alerts) > 0

    def test_no_drift_when_similar_to_baseline(self, tmp_path) -> None:
        """No drift should be detected when metrics match baseline."""
        baseline = {
            "mean_probability": 0.5,
            "std_probability": 0.1,
            "low_confidence_ratio": 0.2,
            "prediction_ratio": {"real": 0.5, "ai_generated": 0.5},
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        detector = DriftDetector(
            window_size=100,
            drift_threshold=0.15,
            baseline_path=baseline_path,
        )

        # Add balanced predictions matching baseline
        for _ in range(50):
            detector.record_prediction(0.45, "real")
            detector.record_prediction(0.55, "ai_generated")

        metrics = detector.check_drift()
        assert metrics.drift_detected is False

    def test_prediction_ratio_drift_alert(self, tmp_path) -> None:
        """Should alert when prediction ratio drifts > 0.2."""
        baseline = {
            "mean_probability": 0.5,
            "std_probability": 0.1,
            "low_confidence_ratio": 0.0,
            "prediction_ratio": {"real": 0.5, "ai_generated": 0.5},
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        detector = DriftDetector(
            window_size=100,
            drift_threshold=0.5,  # High threshold so only ratio triggers
            baseline_path=baseline_path,
        )

        # All AI predictions (ratio drifts from 0.5 to 1.0)
        for _ in range(100):
            detector.record_prediction(0.51, "ai_generated")

        metrics = detector.check_drift()
        assert any("ratio" in a.lower() for a in metrics.alerts)


class TestDriftDetectorGetStatus:
    """Tests for get_status method."""

    def test_status_empty(self) -> None:
        """Status with no data should return sensible defaults."""
        detector = DriftDetector(window_size=100)
        status = detector.get_status()

        assert status["window_size"] == 0
        assert status["window_capacity"] == 100
        assert status["mean_probability"] == 0.5
        assert status["drift_detected"] is False
        assert status["has_baseline"] is False

    def test_status_with_baseline(self, tmp_path) -> None:
        """Status should report has_baseline=True when baseline loaded."""
        baseline = {
            "mean_probability": 0.5,
            "std_probability": 0.1,
            "low_confidence_ratio": 0.1,
            "prediction_ratio": {"real": 0.5, "ai_generated": 0.5},
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        detector = DriftDetector(baseline_path=baseline_path)
        status = detector.get_status()
        assert status["has_baseline"] is True

    def test_status_values_rounded(self) -> None:
        """Status values should be rounded to 4 decimal places."""
        detector = DriftDetector(window_size=10)
        for _ in range(10):
            detector.record_prediction(0.123456789, "real")

        status = detector.get_status()
        mean_str = str(status["mean_probability"])
        # Should have at most 4 decimal places
        if "." in mean_str:
            assert len(mean_str.split(".")[1]) <= 4


class TestDriftMetricsDataclass:
    """Tests for DriftMetrics dataclass."""

    def test_default_alerts(self) -> None:
        """Default alerts should be empty list."""
        metrics = DriftMetrics(
            timestamp="2025-01-01T00:00:00",
            window_size=0,
            mean_probability=0.5,
            std_probability=0.0,
            low_confidence_ratio=0.0,
            prediction_ratio={"real": 0.5, "ai_generated": 0.5},
            drift_detected=False,
            drift_score=0.0,
        )
        assert metrics.alerts == []

    def test_custom_alerts(self) -> None:
        """Alerts should accept custom list."""
        metrics = DriftMetrics(
            timestamp="2025-01-01T00:00:00",
            window_size=100,
            mean_probability=0.8,
            std_probability=0.1,
            low_confidence_ratio=0.1,
            prediction_ratio={"real": 0.2, "ai_generated": 0.8},
            drift_detected=True,
            drift_score=0.3,
            alerts=["Probability drift: 0.300"],
        )
        assert len(metrics.alerts) == 1
