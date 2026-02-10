"""Tests for Prometheus metrics module."""

from src.monitoring.metrics import (
    record_prediction,
    set_app_info,
    set_model_info,
)


class TestSetAppInfo:
    """Tests for set_app_info function."""

    def test_set_app_info_default_env(self) -> None:
        """Should set app info without error."""
        set_app_info("1.0.0")

    def test_set_app_info_custom_env(self) -> None:
        """Should accept custom environment."""
        set_app_info("2.0.0", environment="staging")


class TestSetModelInfo:
    """Tests for set_model_info function."""

    def test_set_model_info(self) -> None:
        """Should set model info without error."""
        set_model_info(
            name="test-model",
            version="1.0.0",
            architecture="efficientnet",
            parameters=1000000,
        )


class TestRecordPrediction:
    """Tests for record_prediction function."""

    def test_record_successful_prediction(self) -> None:
        """Should record a successful prediction."""
        record_prediction(
            prediction="ai_generated",
            probability=0.85,
            confidence="high",
            latency_seconds=0.05,
            success=True,
        )

    def test_record_failed_prediction(self) -> None:
        """Should record a failed prediction."""
        record_prediction(
            prediction="none",
            probability=0.0,
            confidence="none",
            latency_seconds=0.0,
            success=False,
        )

    def test_record_real_prediction(self) -> None:
        """Should record a real prediction."""
        record_prediction(
            prediction="real",
            probability=0.15,
            confidence="high",
            latency_seconds=0.03,
            success=True,
        )
