"""Tests for Grad-CAM explainability module."""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.inference.explainer import GradCAMExplainer, _rebuild_classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_test_image(size: tuple[int, int] = (224, 224), color: str = "red") -> bytes:
    """Create a simple JPEG image in memory."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests: _rebuild_classifier
# ---------------------------------------------------------------------------
class TestRebuildClassifier:
    """Tests for classifier reconstruction from checkpoint keys."""

    def test_old_checkpoint_no_batchnorm(self) -> None:
        """Old checkpoints have no BatchNorm1d in classifier."""
        state_dict = {
            "classifier.0.weight": torch.randn(512, 1280),
            "classifier.0.bias": torch.randn(512),
            "classifier.3.weight": torch.randn(1, 512),
            "classifier.3.bias": torch.randn(1),
        }
        classifier = _rebuild_classifier(state_dict, feature_dim=1280, dropout=0.3)
        # Should be: Linear, ReLU, Dropout, Linear → 4 layers
        assert len(classifier) == 4
        assert isinstance(classifier[0], torch.nn.Linear)
        assert isinstance(classifier[3], torch.nn.Linear)

    def test_new_checkpoint_with_batchnorm(self) -> None:
        """New checkpoints include BatchNorm1d."""
        state_dict = {
            "classifier.0.weight": torch.randn(512, 1280),
            "classifier.0.bias": torch.randn(512),
            "classifier.1.weight": torch.randn(512),
            "classifier.1.bias": torch.randn(512),
            "classifier.1.running_mean": torch.zeros(512),
            "classifier.1.running_var": torch.ones(512),
            "classifier.4.weight": torch.randn(1, 512),
            "classifier.4.bias": torch.randn(1),
        }
        classifier = _rebuild_classifier(state_dict, feature_dim=1280, dropout=0.3)
        # Should be: Linear, BatchNorm, ReLU, Dropout, Linear → 5 layers
        assert len(classifier) == 5
        assert isinstance(classifier[1], torch.nn.BatchNorm1d)


# ---------------------------------------------------------------------------
# Integration tests: GradCAMExplainer
# ---------------------------------------------------------------------------
class TestGradCAMExplainer:
    """Tests for the GradCAMExplainer class."""

    @pytest.fixture()
    def explainer(self) -> GradCAMExplainer:
        """Create an explainer with the real model checkpoint."""
        return GradCAMExplainer("models/checkpoints/best_model.pt")

    def test_is_ready(self, explainer: GradCAMExplainer) -> None:
        assert explainer.is_ready()

    def test_model_version_set(self, explainer: GradCAMExplainer) -> None:
        assert explainer.model_version.startswith("1.0.")

    def test_explain_returns_required_keys(self, explainer: GradCAMExplainer) -> None:
        image_bytes = _make_test_image()
        result = explainer.explain(image_bytes)

        assert "prediction" in result
        assert "probability" in result
        assert "confidence" in result
        assert "heatmap_base64" in result
        assert "inference_time_ms" in result

    def test_explain_prediction_values(self, explainer: GradCAMExplainer) -> None:
        result = explainer.explain(_make_test_image())

        assert result["prediction"] in ("real", "ai_generated")
        assert 0.0 <= result["probability"] <= 1.0
        assert result["confidence"] in ("low", "medium", "high")
        assert result["inference_time_ms"] > 0

    def test_explain_heatmap_is_valid_jpeg(self, explainer: GradCAMExplainer) -> None:
        result = explainer.explain(_make_test_image())
        heatmap_bytes = base64.b64decode(result["heatmap_base64"])

        # Should be loadable as a JPEG image
        img = Image.open(io.BytesIO(heatmap_bytes))
        assert img.format == "JPEG"
        assert img.size == (224, 224)

    def test_explain_different_image_sizes(self, explainer: GradCAMExplainer) -> None:
        """Explainer should handle various image sizes."""
        for size in [(100, 100), (640, 480), (1920, 1080)]:
            result = explainer.explain(_make_test_image(size=size))
            assert result["heatmap_base64"]

    def test_explain_different_colors(self, explainer: GradCAMExplainer) -> None:
        """Different inputs should produce valid results."""
        for color in ("red", "green", "blue", "white", "black"):
            result = explainer.explain(_make_test_image(color=color))
            assert result["prediction"] in ("real", "ai_generated")


class TestGradCAMExplainerNotLoaded:
    """Tests for failure modes."""

    def test_missing_model_path(self) -> None:
        explainer = GradCAMExplainer("nonexistent/model.pt")
        assert not explainer.is_ready()

    def test_explain_raises_when_not_ready(self) -> None:
        explainer = GradCAMExplainer("nonexistent/model.pt")
        with pytest.raises(RuntimeError, match="Explainer not loaded"):
            explainer.explain(_make_test_image())
