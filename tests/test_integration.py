"""Integration tests — end-to-end API testing with TestClient."""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


def _make_image_bytes(size=(224, 224), color="red", fmt="JPEG") -> bytes:
    """Create image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def _create_fake_checkpoint(path: Path) -> Path:
    """Create a fake model checkpoint."""
    from src.training.model import create_model

    model = create_model(pretrained=False)
    checkpoint = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_accuracy": 0.92,
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


@pytest.fixture(scope="module")
def model_checkpoint(tmp_path_factory) -> Path:
    """Create a shared fake checkpoint for integration tests."""
    path = tmp_path_factory.mktemp("models") / "best_model.pt"
    return _create_fake_checkpoint(path)


@pytest.fixture()
def loaded_client(model_checkpoint: Path) -> TestClient:
    """Create a TestClient with a loaded model.

    Patches the model path so the API actually loads the checkpoint.
    """
    with patch.dict(os.environ, {"MODEL_PATH": str(model_checkpoint)}):
        # Force reimport so the lifespan picks up the new env
        import importlib
        import src.inference.api
        importlib.reload(src.inference.api)
        from src.inference.api import app

        with TestClient(app) as client:
            yield client


class TestEndToEndHealth:
    """Integration tests for /health endpoint."""

    def test_health_healthy_with_model(self, loaded_client: TestClient) -> None:
        """Health should be healthy when model is loaded."""
        response = loaded_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "1.0.3"
        assert data["uptime_seconds"] >= 0


class TestEndToEndPredict:
    """Integration tests for /predict endpoint."""

    def test_predict_jpeg_image(self, loaded_client: TestClient) -> None:
        """Should return a valid prediction for a JPEG image."""
        image_bytes = _make_image_bytes(fmt="JPEG")
        response = loaded_client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["prediction"] in ("real", "ai_generated")
        assert 0.0 <= data["probability"] <= 1.0
        assert data["confidence"] in ("low", "medium", "high")
        assert data["inference_time_ms"] >= 0
        assert data["model_version"] == "1.0.3"

    def test_predict_png_image(self, loaded_client: TestClient) -> None:
        """Should return a valid prediction for a PNG image."""
        image_bytes = _make_image_bytes(fmt="PNG")
        response = loaded_client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("real", "ai_generated")

    def test_predict_rejects_text_file(self, loaded_client: TestClient) -> None:
        """Should reject non-image files with 400."""
        response = loaded_client.post(
            "/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_rejects_oversized_file(self, loaded_client: TestClient) -> None:
        """Should reject files over 10MB with 413."""
        large_data = b"x" * (11 * 1024 * 1024)
        response = loaded_client.post(
            "/predict",
            files={"file": ("big.jpg", large_data, "image/jpeg")},
        )
        assert response.status_code == 413


class TestEndToEndBatch:
    """Integration tests for /predict/batch endpoint."""

    def test_batch_single_image(self, loaded_client: TestClient) -> None:
        """Batch with single image should work."""
        image_bytes = _make_image_bytes()
        response = loaded_client.post(
            "/predict/batch",
            files=[("files", ("test.jpg", image_bytes, "image/jpeg"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["successful"] == 1
        assert data["failed"] == 0

    def test_batch_multiple_images(self, loaded_client: TestClient) -> None:
        """Batch with multiple images should return results for each."""
        image_bytes = _make_image_bytes()
        files = [
            ("files", (f"test{i}.jpg", image_bytes, "image/jpeg"))
            for i in range(3)
        ]
        response = loaded_client.post("/predict/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["successful"] == 3

    def test_batch_mixed_valid_invalid(self, loaded_client: TestClient) -> None:
        """Batch should handle mixed valid and invalid files."""
        image_bytes = _make_image_bytes()
        files = [
            ("files", ("good.jpg", image_bytes, "image/jpeg")),
            ("files", ("bad.txt", b"not image", "text/plain")),
        ]
        response = loaded_client.post("/predict/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["successful"] == 1
        assert data["failed"] == 1

    def test_batch_too_many_files(self, loaded_client: TestClient) -> None:
        """Batch should reject more than 20 files."""
        image_bytes = _make_image_bytes()
        files = [
            ("files", (f"test{i}.jpg", image_bytes, "image/jpeg"))
            for i in range(25)
        ]
        response = loaded_client.post("/predict/batch", files=files)
        assert response.status_code == 400


class TestEndToEndMiscEndpoints:
    """Integration tests for other endpoints."""

    def test_root_endpoint(self, loaded_client: TestClient) -> None:
        """Root endpoint should return API info."""
        response = loaded_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI Product Photo Detector"
        assert data["version"] == "1.0.0"

    def test_metrics_endpoint(self, loaded_client: TestClient) -> None:
        """Metrics endpoint should return Prometheus data."""
        response = loaded_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_drift_endpoint(self, loaded_client: TestClient) -> None:
        """Drift endpoint should return status."""
        response = loaded_client.get("/drift")
        assert response.status_code == 200
        data = response.json()
        assert "window_size" in data
        assert "drift_detected" in data
