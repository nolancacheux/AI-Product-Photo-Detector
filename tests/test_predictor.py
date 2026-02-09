"""Tests for the Predictor class."""

import pytest

from src.inference.predictor import Predictor
from src.inference.schemas import ConfidenceLevel


class TestPredictorInit:
    """Tests for Predictor initialization."""

    def test_non_existent_model_not_ready(self, tmp_path) -> None:
        """Predictor with non-existent path should not be ready."""
        predictor = Predictor(model_path=tmp_path / "nonexistent.pt", device="cpu")
        assert predictor.is_ready() is False

    def test_model_is_none_when_not_loaded(self, tmp_path) -> None:
        """Model attribute should be None when path doesn't exist."""
        predictor = Predictor(model_path=tmp_path / "missing.pt", device="cpu")
        assert predictor.model is None


class TestGetConfidenceLevel:
    """Tests for _get_confidence_level method."""

    @pytest.fixture()
    def predictor(self, tmp_path) -> Predictor:
        """Create a predictor instance (model won't load)."""
        return Predictor(model_path=tmp_path / "fake.pt", device="cpu")

    def test_high_confidence_ai(self, predictor: Predictor) -> None:
        """Probability near 1.0 should give HIGH confidence."""
        assert predictor._get_confidence_level(0.95) == ConfidenceLevel.HIGH

    def test_high_confidence_real(self, predictor: Predictor) -> None:
        """Probability near 0.0 should give HIGH confidence."""
        assert predictor._get_confidence_level(0.05) == ConfidenceLevel.HIGH

    def test_low_confidence_near_boundary(self, predictor: Predictor) -> None:
        """Probability near 0.5 should give LOW confidence."""
        assert predictor._get_confidence_level(0.48) == ConfidenceLevel.LOW

    def test_medium_confidence(self, predictor: Predictor) -> None:
        """Probability between thresholds should give MEDIUM confidence."""
        assert predictor._get_confidence_level(0.75) == ConfidenceLevel.MEDIUM

    def test_exact_boundary_high(self, predictor: Predictor) -> None:
        """Probability at high threshold boundary."""
        result = predictor._get_confidence_level(0.85)
        assert result in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)


class TestPredictFromBytes:
    """Tests for predict_from_bytes method."""

    def test_raises_runtime_error_when_not_loaded(self, tmp_path) -> None:
        """Should raise RuntimeError when model is not loaded."""
        predictor = Predictor(model_path=tmp_path / "fake.pt", device="cpu")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict_from_bytes(b"fake image data")


class TestPredictFromPath:
    """Tests for predict_from_path method."""

    def test_raises_file_not_found_for_missing_file(self, tmp_path) -> None:
        """Should raise FileNotFoundError for non-existent image."""
        predictor = Predictor(model_path=tmp_path / "fake.pt", device="cpu")
        with pytest.raises(FileNotFoundError, match="Image not found"):
            predictor.predict_from_path(tmp_path / "missing_image.jpg")
