"""Tests for the drift detection module."""

from src.monitoring.drift import DriftDetector, DriftMetrics


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_detector_initialization(self) -> None:
        """Test detector can be initialized."""
        detector = DriftDetector(window_size=100)
        assert detector.window_size == 100
        assert len(detector.predictions) == 0

    def test_record_prediction(self) -> None:
        """Test recording predictions."""
        detector = DriftDetector(window_size=10)

        detector.record_prediction(0.8, "ai_generated")
        detector.record_prediction(0.2, "real")

        assert len(detector.predictions) == 2

    def test_window_size_limit(self) -> None:
        """Test sliding window respects size limit."""
        detector = DriftDetector(window_size=5)

        for _i in range(10):
            detector.record_prediction(0.5, "ai_generated")

        assert len(detector.predictions) == 5

    def test_compute_metrics_empty(self) -> None:
        """Test metrics computation with no data."""
        detector = DriftDetector()
        metrics = detector.check_drift()

        assert isinstance(metrics, DriftMetrics)
        assert metrics.window_size == 0
        assert not metrics.drift_detected

    def test_compute_metrics_with_data(self) -> None:
        """Test metrics computation with predictions."""
        detector = DriftDetector(window_size=100)

        # Add balanced predictions
        for _ in range(50):
            detector.record_prediction(0.8, "ai_generated")
            detector.record_prediction(0.2, "real")

        metrics = detector.check_drift()

        assert metrics.window_size == 100
        assert 0.4 < metrics.mean_probability < 0.6
        assert metrics.prediction_ratio["ai_generated"] == 0.5
        assert metrics.prediction_ratio["real"] == 0.5

    def test_low_confidence_detection(self) -> None:
        """Test detection of low confidence predictions."""
        detector = DriftDetector(window_size=100, confidence_threshold=0.3)

        # Add predictions near 0.5 (low confidence)
        for _ in range(100):
            detector.record_prediction(0.5, "ai_generated")

        metrics = detector.check_drift()

        # All predictions should be low confidence
        assert metrics.low_confidence_ratio == 1.0

    def test_drift_detection_without_baseline(self) -> None:
        """Test drift detection without baseline."""
        detector = DriftDetector()

        for _ in range(50):
            detector.record_prediction(0.9, "ai_generated")

        metrics = detector.check_drift()

        # Without baseline, no drift should be detected
        assert not metrics.drift_detected
        assert metrics.drift_score == 0.0

    def test_get_status(self) -> None:
        """Test status reporting."""
        detector = DriftDetector(window_size=100)

        for _ in range(50):
            detector.record_prediction(0.7, "ai_generated")

        status = detector.get_status()

        assert status["window_size"] == 50
        assert status["window_capacity"] == 100
        assert "mean_probability" in status
        assert "drift_detected" in status
        assert status["has_baseline"] is False
