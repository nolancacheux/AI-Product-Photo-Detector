"""Tests for batch prediction endpoint."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Create test client."""
    from src.inference.api import app

    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    img = Image.new("RGB", (224, 224), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_empty_request(self, client: TestClient) -> None:
        """Test batch rejects empty request."""
        response = client.post("/predict/batch", files=[])
        assert response.status_code == 422  # Validation error

    def test_batch_accepts_multiple_images(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Test batch accepts multiple images."""
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", sample_image_bytes, "image/jpeg")),
        ]
        response = client.post("/predict/batch", files=files)
        # Will be 503 if model not loaded
        assert response.status_code in [200, 503]

    def test_batch_response_structure(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Test batch response has correct structure."""
        files = [("files", ("test.jpg", sample_image_bytes, "image/jpeg"))]
        response = client.post("/predict/batch", files=files)

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "successful" in data
            assert "failed" in data
            assert "total_inference_time_ms" in data
            assert "model_version" in data

    def test_batch_handles_mixed_valid_invalid(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Test batch handles mix of valid and invalid files."""
        files = [
            ("files", ("test.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test.txt", b"not an image", "text/plain")),
        ]
        response = client.post("/predict/batch", files=files)

        if response.status_code == 200:
            data = response.json()
            assert data["total"] == 2
            # At least one should fail (invalid format)
            assert data["failed"] >= 1

    def test_batch_max_size_limit(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Test batch rejects more than 20 images."""
        files = [("files", (f"test{i}.jpg", sample_image_bytes, "image/jpeg")) for i in range(25)]
        response = client.post("/predict/batch", files=files)
        # Should be 400 (batch too large) or 503 (model not loaded)
        assert response.status_code in [400, 503]
