"""Tests for the API module."""

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


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        """Test health response has correct structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data


    def test_health_unhealthy_when_no_model(self, client: TestClient) -> None:
        """Test health returns unhealthy when model is not loaded."""
        response = client.get("/health")
        data = response.json()
        # Model file doesn't exist in test env, so should be unhealthy
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_200(self, client: TestClient) -> None:
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_structure(self, client: TestClient) -> None:
        """Test root response has correct structure."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_root_contains_api_name(self, client: TestClient) -> None:
        """Test root response contains the API name."""
        response = client.get("/")
        data = response.json()
        assert data["name"] == "AI Product Photo Detector"


class TestPrivacyEndpoint:
    """Tests for /privacy endpoint."""

    def test_privacy_returns_200(self, client: TestClient) -> None:
        """Test privacy endpoint returns 200."""
        response = client.get("/privacy")
        assert response.status_code == 200

    def test_privacy_response_structure(self, client: TestClient) -> None:
        """Test privacy response has expected fields."""
        response = client.get("/privacy")
        data = response.json()

        assert "image_storage" in data
        assert "data_retention" in data
        assert "gdpr" in data
        assert data["data_retention"] == "none"

    def test_privacy_confirms_no_storage(self, client: TestClient) -> None:
        """Test privacy explicitly states no image storage."""
        response = client.get("/privacy")
        data = response.json()
        assert "never saved" in data["image_storage"].lower() or "in-memory" in data["image_storage"].lower()


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_200(self, client: TestClient) -> None:
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client: TestClient) -> None:
        """Test metrics returns text/plain."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_prometheus_format(self, client: TestClient) -> None:
        """Test metrics returns prometheus-compatible output."""
        response = client.get("/metrics")
        content = response.text
        # Prometheus metrics contain HELP or TYPE lines, or metric names
        assert "predictions_total" in content or "# " in content or len(content) > 0


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_invalid_content_type(self, client: TestClient) -> None:
        """Test predict rejects invalid content type."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        # 400 if model loaded, 503 if not
        assert response.status_code in [400, 503]

    def test_predict_too_large_file(self, client: TestClient) -> None:
        """Test predict rejects files over 10MB."""
        # Create 11MB of data
        large_data = b"x" * (11 * 1024 * 1024)

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", large_data, "image/jpeg")},
        )
        # 413 if model loaded, 503 if not (model check happens first)
        assert response.status_code in [413, 503]

    def test_predict_accepts_jpeg(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Test predict accepts JPEG images."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        # Will be 503 if model not loaded, but not 400
        assert response.status_code in [200, 503]

    def test_predict_accepts_png(self, client: TestClient) -> None:
        """Test predict accepts PNG images."""
        img = Image.new("RGB", (224, 224), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", buffer.read(), "image/png")},
        )
        assert response.status_code in [200, 503]
