"""Tests for the shadow/comparison module."""

from unittest.mock import MagicMock, patch

import pytest

from src.inference.schemas import (
    ConfidenceLevel,
    PredictionResult,
    PredictResponse,
)


def _make_predict_response(
    prediction: PredictionResult = PredictionResult.AI_GENERATED,
    probability: float = 0.9,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> PredictResponse:
    return PredictResponse(
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        inference_time_ms=10.0,
        model_version="1.0.0",
    )


@pytest.fixture(autouse=True)
def _reset_shadow_globals():
    """Reset module-level globals before each test."""
    import src.inference.shadow as mod

    mod._shadow_predictor = None
    mod._shadow_load_attempted = False
    yield
    mod._shadow_predictor = None
    mod._shadow_load_attempted = False


class TestGetShadowPredictor:
    """Tests for get_shadow_predictor()."""

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_env_not_set(self) -> None:
        from src.inference.shadow import get_shadow_predictor

        result = get_shadow_predictor()
        assert result is None

    @patch.dict("os.environ", {"SHADOW_MODEL_PATH": "/nonexistent/model.pt"})
    @patch("src.inference.shadow.Predictor")
    def test_returns_none_with_invalid_path(self, mock_predictor_cls: MagicMock) -> None:
        mock_predictor_cls.side_effect = FileNotFoundError("not found")

        from src.inference.shadow import get_shadow_predictor

        result = get_shadow_predictor()
        assert result is None

    @patch.dict("os.environ", {"SHADOW_MODEL_PATH": "/some/model.pt"})
    @patch("src.inference.shadow.Predictor")
    def test_returns_predictor_when_ready(self, mock_predictor_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.is_ready.return_value = True
        mock_predictor_cls.return_value = mock_instance

        from src.inference.shadow import get_shadow_predictor

        result = get_shadow_predictor()
        assert result is mock_instance
        mock_predictor_cls.assert_called_once_with(model_path="/some/model.pt")

    @patch.dict("os.environ", {"SHADOW_MODEL_PATH": "/some/model.pt"})
    @patch("src.inference.shadow.Predictor")
    def test_returns_none_when_not_ready(self, mock_predictor_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.is_ready.return_value = False
        mock_predictor_cls.return_value = mock_instance

        from src.inference.shadow import get_shadow_predictor

        result = get_shadow_predictor()
        assert result is None

    @patch.dict("os.environ", {"SHADOW_MODEL_PATH": "/some/model.pt"})
    @patch("src.inference.shadow.Predictor")
    def test_caches_result_on_second_call(self, mock_predictor_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.is_ready.return_value = True
        mock_predictor_cls.return_value = mock_instance

        from src.inference.shadow import get_shadow_predictor

        first = get_shadow_predictor()
        second = get_shadow_predictor()
        assert first is second
        assert mock_predictor_cls.call_count == 1


class TestComparePredictions:
    """Tests for compare_predictions()."""

    def test_no_shadow_returns_agreement(self) -> None:
        from src.inference.shadow import compare_predictions

        primary = _make_predict_response()
        result = compare_predictions(primary, None)

        assert result.agreement is True
        assert result.difference == 0.0
        assert result.shadow is None
        assert result.message == "No shadow model configured"

    def test_matching_predictions_agreement(self) -> None:
        from src.inference.shadow import compare_predictions

        primary = _make_predict_response(
            prediction=PredictionResult.AI_GENERATED,
            probability=0.9,
        )
        shadow = _make_predict_response(
            prediction=PredictionResult.AI_GENERATED,
            probability=0.85,
        )

        result = compare_predictions(primary, shadow)
        assert result.agreement is True
        assert result.difference == pytest.approx(0.05, abs=1e-4)
        assert result.message is None

    def test_different_predictions_disagreement(self) -> None:
        from src.inference.shadow import compare_predictions

        primary = _make_predict_response(
            prediction=PredictionResult.AI_GENERATED,
            probability=0.9,
        )
        shadow = _make_predict_response(
            prediction=PredictionResult.REAL,
            probability=0.2,
        )

        result = compare_predictions(primary, shadow)
        assert result.agreement is False
        assert result.difference == pytest.approx(0.7, abs=1e-4)

    def test_difference_is_absolute(self) -> None:
        from src.inference.shadow import compare_predictions

        primary = _make_predict_response(probability=0.3)
        shadow = _make_predict_response(probability=0.8)

        result = compare_predictions(primary, shadow)
        assert result.difference == pytest.approx(0.5, abs=1e-4)

    def test_primary_is_always_returned(self) -> None:
        from src.inference.shadow import compare_predictions

        primary = _make_predict_response()
        shadow = _make_predict_response()

        result = compare_predictions(primary, shadow)
        assert result.primary is primary
        assert result.shadow is shadow
