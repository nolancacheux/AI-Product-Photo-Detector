"""Extended tests for the Predictor class with mocked model."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from src.inference.predictor import Predictor
from src.inference.schemas import ConfidenceLevel, PredictionResult


def _make_image_bytes(size=(224, 224), color="red", fmt="JPEG") -> bytes:
    """Create image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def _create_fake_checkpoint(path: Path, probability: float = 0.9) -> Path:
    """Create a fake model checkpoint that loads correctly."""
    from src.training.model import create_model

    model = create_model(pretrained=False)
    checkpoint = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_accuracy": 0.95,
        "config": {
            "model": {
                "name": "efficientnet_b0",
                "dropout": 0.3,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class TestPredictorWithModel:
    """Tests for Predictor with a real (untrained) model."""

    @pytest.fixture()
    def model_path(self, tmp_path) -> Path:
        """Create a fake checkpoint and return its path."""
        return _create_fake_checkpoint(tmp_path / "model.pt")

    @pytest.fixture()
    def predictor(self, model_path) -> Predictor:
        """Create a predictor with a loaded model."""
        return Predictor(model_path=model_path, device="cpu")

    def test_model_loads_successfully(self, predictor: Predictor) -> None:
        """Predictor should load the checkpoint model."""
        assert predictor.is_ready() is True
        assert predictor.model is not None

    def test_model_version_from_checkpoint(self, predictor: Predictor) -> None:
        """Model version should be derived from checkpoint epoch."""
        assert predictor.model_version == "1.0.5"

    def test_predict_from_bytes_returns_response(self, predictor: Predictor) -> None:
        """predict_from_bytes should return a PredictResponse."""
        image_bytes = _make_image_bytes()
        response = predictor.predict_from_bytes(image_bytes)

        assert response.prediction in (PredictionResult.REAL, PredictionResult.AI_GENERATED)
        assert 0.0 <= response.probability <= 1.0
        assert response.confidence in (
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
        )
        assert response.inference_time_ms >= 0
        assert response.model_version == "1.0.5"

    def test_predict_from_bytes_png(self, predictor: Predictor) -> None:
        """predict_from_bytes should handle PNG images."""
        image_bytes = _make_image_bytes(fmt="PNG")
        response = predictor.predict_from_bytes(image_bytes)
        assert 0.0 <= response.probability <= 1.0

    def test_predict_from_bytes_small_image(self, predictor: Predictor) -> None:
        """predict_from_bytes should handle small images (resized internally)."""
        image_bytes = _make_image_bytes(size=(32, 32))
        response = predictor.predict_from_bytes(image_bytes)
        assert 0.0 <= response.probability <= 1.0

    def test_predict_from_bytes_large_image(self, predictor: Predictor) -> None:
        """predict_from_bytes should handle large images."""
        image_bytes = _make_image_bytes(size=(1024, 768))
        response = predictor.predict_from_bytes(image_bytes)
        assert 0.0 <= response.probability <= 1.0

    def test_predict_from_bytes_invalid_data(self, predictor: Predictor) -> None:
        """predict_from_bytes should raise ValueError for invalid data."""
        with pytest.raises(ValueError, match="Failed to process image"):
            predictor.predict_from_bytes(b"not an image at all")

    def test_predict_from_path_success(self, predictor: Predictor, tmp_path) -> None:
        """predict_from_path should work with a real file."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(img_path)

        response = predictor.predict_from_path(img_path)
        assert 0.0 <= response.probability <= 1.0

    def test_predict_from_path_missing_file(self, predictor: Predictor, tmp_path) -> None:
        """predict_from_path should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            predictor.predict_from_path(tmp_path / "nonexistent.jpg")

    def test_predict_from_path_string_path(self, predictor: Predictor, tmp_path) -> None:
        """predict_from_path should accept string paths."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(img_path)

        response = predictor.predict_from_path(str(img_path))
        assert response.prediction in (PredictionResult.REAL, PredictionResult.AI_GENERATED)


class TestPredictorDeviceSelection:
    """Tests for device auto-selection logic."""

    def test_cpu_device(self, tmp_path) -> None:
        """Explicit CPU device should work."""
        predictor = Predictor(model_path=tmp_path / "fake.pt", device="cpu")
        assert str(predictor.device) == "cpu"

    def test_auto_device_fallback_cpu(self, tmp_path) -> None:
        """Auto device should fallback to CPU when no GPU."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            predictor = Predictor(model_path=tmp_path / "fake.pt", device="auto")
            assert str(predictor.device) == "cpu"

    def test_auto_device_cuda(self, tmp_path) -> None:
        """Auto device should select CUDA when available."""
        with patch("torch.cuda.is_available", return_value=True):
            predictor = Predictor(model_path=tmp_path / "fake.pt", device="auto")
            assert str(predictor.device) == "cuda"


class TestPredictorThresholds:
    """Tests for custom threshold configurations."""

    @pytest.fixture()
    def model_path(self, tmp_path) -> Path:
        return _create_fake_checkpoint(tmp_path / "model.pt")

    def test_custom_threshold(self, model_path) -> None:
        """Custom classification threshold should work."""
        predictor = Predictor(model_path=model_path, device="cpu", threshold=0.7)
        assert predictor.threshold == 0.7

    def test_custom_confidence_thresholds(self, model_path) -> None:
        """Custom confidence thresholds should work."""
        predictor = Predictor(
            model_path=model_path,
            device="cpu",
            high_confidence_threshold=0.9,
            low_confidence_threshold=0.2,
        )
        assert predictor.high_confidence_threshold == 0.9
        assert predictor.low_confidence_threshold == 0.2


class TestGetConfidenceLevelExtended:
    """Extended tests for _get_confidence_level edge cases."""

    @pytest.fixture()
    def predictor(self, tmp_path) -> Predictor:
        return Predictor(model_path=tmp_path / "fake.pt", device="cpu")

    def test_probability_zero(self, predictor: Predictor) -> None:
        """Probability 0.0 should give HIGH confidence."""
        assert predictor._get_confidence_level(0.0) == ConfidenceLevel.HIGH

    def test_probability_one(self, predictor: Predictor) -> None:
        """Probability 1.0 should give HIGH confidence."""
        assert predictor._get_confidence_level(1.0) == ConfidenceLevel.HIGH

    def test_probability_half(self, predictor: Predictor) -> None:
        """Probability 0.5 should give LOW confidence."""
        assert predictor._get_confidence_level(0.5) == ConfidenceLevel.LOW

    def test_probability_just_above_low(self, predictor: Predictor) -> None:
        """Probability slightly outside low band should be MEDIUM."""
        # Default: low_confidence_threshold = 0.3, so distance < 0.2 → LOW
        # Distance from 0.5 at 0.29 = 0.21 → MEDIUM
        assert predictor._get_confidence_level(0.29) == ConfidenceLevel.MEDIUM

    def test_probability_just_below_high(self, predictor: Predictor) -> None:
        """Probability just below high threshold should be MEDIUM."""
        assert predictor._get_confidence_level(0.79) == ConfidenceLevel.MEDIUM


class TestPredictorLoadModelFailure:
    """Tests for model loading failure cases."""

    def test_corrupt_checkpoint(self, tmp_path) -> None:
        """Corrupt checkpoint should leave model as None."""
        corrupt_path = tmp_path / "corrupt.pt"
        corrupt_path.write_bytes(b"this is not a valid checkpoint")

        predictor = Predictor(model_path=corrupt_path, device="cpu")
        assert predictor.model is None
        assert predictor.is_ready() is False
