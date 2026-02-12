"""Tests for the Streamlit UI module (src/ui/app.py)."""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")


def _make_jpeg(size: tuple[int, int] = (100, 100), color: str = "red") -> bytes:
    """Create JPEG bytes for testing."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png_rgba(size: tuple[int, int] = (100, 100)) -> bytes:
    """Create RGBA PNG bytes for testing."""
    img = Image.new("RGBA", size, (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCompressImage:
    """Tests for compress_image()."""

    def test_small_image_returned_unchanged(self) -> None:
        from src.ui.app import compress_image

        data = _make_jpeg()
        result = compress_image(data, max_size_mb=10.0)
        assert result == data

    def test_large_image_is_compressed(self) -> None:
        from src.ui.app import compress_image

        # Create a large image that exceeds 0.001 MB threshold
        large = _make_jpeg(size=(3000, 3000))
        result = compress_image(large, max_size_mb=0.001)
        assert len(result) < len(large)

    def test_rgba_converted_to_rgb(self) -> None:
        from src.ui.app import compress_image

        rgba_data = _make_png_rgba(size=(3000, 3000))
        result = compress_image(rgba_data, max_size_mb=0.001)
        # Result should be valid JPEG
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_oversized_image_resized(self) -> None:
        from src.ui.app import compress_image

        # Image larger than max_dim=2048
        huge = _make_jpeg(size=(4000, 4000))
        result = compress_image(huge, max_size_mb=0.001)
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= 2048


class TestCheckApiHealth:
    """Tests for check_api_health()."""

    @patch("src.ui.app.httpx.get")
    def test_healthy_api(self, mock_get: MagicMock) -> None:
        from src.ui.app import check_api_health

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "model_version": "1.0"}
        mock_get.return_value = mock_response

        result = check_api_health()
        assert result is not None
        assert result["status"] == "healthy"

    @patch("src.ui.app.httpx.get")
    def test_unhealthy_api_returns_none(self, mock_get: MagicMock) -> None:
        from src.ui.app import check_api_health

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = check_api_health()
        assert result is None

    @patch("src.ui.app.httpx.get", side_effect=ConnectionError("refused"))
    def test_connection_error_returns_none(self, mock_get: MagicMock) -> None:
        from src.ui.app import check_api_health

        result = check_api_health()
        assert result is None


class TestPredictImage:
    """Tests for predict_image()."""

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post")
    def test_successful_prediction(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import predict_image

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "real",
            "probability": 0.2,
        }
        mock_post.return_value = mock_response

        result = predict_image(_make_jpeg(), "test.jpg")
        assert result is not None
        assert result["prediction"] == "real"

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post")
    def test_api_error_shows_error(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import predict_image

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Bad request"}
        mock_post.return_value = mock_response

        result = predict_image(_make_jpeg(), "test.jpg")
        assert result is None
        mock_st.error.assert_called_once()

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post")
    def test_api_error_non_json_response(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import predict_image

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("not JSON")
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        result = predict_image(_make_jpeg(), "test.jpg")
        assert result is None

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post", side_effect=ConnectionError("refused"))
    def test_connection_error(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import predict_image

        result = predict_image(_make_jpeg(), "test.jpg")
        assert result is None
        mock_st.error.assert_called_once()


class TestExplainImage:
    """Tests for explain_image()."""

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post")
    def test_successful_explanation(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import explain_image

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prediction": "ai_generated",
            "heatmap_base64": "abc123",
        }
        mock_post.return_value = mock_response

        result = explain_image(_make_jpeg(), "test.jpg")
        assert result is not None
        assert result["heatmap_base64"] == "abc123"

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post")
    def test_explain_api_failure(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import explain_image

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = explain_image(_make_jpeg(), "test.jpg")
        assert result is None

    @patch("src.ui.app.st")
    @patch("src.ui.app.httpx.post", side_effect=Exception("network error"))
    def test_explain_connection_error(self, mock_post: MagicMock, mock_st: MagicMock) -> None:
        from src.ui.app import explain_image

        result = explain_image(_make_jpeg(), "test.jpg")
        assert result is None


class TestDisplayResult:
    """Tests for display_result()."""

    @patch("src.ui.app.st")
    def test_display_ai_generated(self, mock_st: MagicMock) -> None:
        from src.ui.app import display_result

        display_result(
            {
                "prediction": "ai_generated",
                "probability": 0.95,
                "inference_time_ms": 42.0,
            }
        )
        mock_st.markdown.assert_called()

    @patch("src.ui.app.st")
    def test_display_real(self, mock_st: MagicMock) -> None:
        from src.ui.app import display_result

        display_result(
            {
                "prediction": "real",
                "probability": 0.1,
                "inference_time_ms": 30.0,
            }
        )
        mock_st.markdown.assert_called()

    @patch("src.ui.app.st")
    def test_confidence_levels(self, mock_st: MagicMock) -> None:
        from src.ui.app import display_result

        # Very high confidence
        display_result({"prediction": "ai_generated", "probability": 0.99, "inference_time_ms": 10})
        # High confidence
        display_result({"prediction": "ai_generated", "probability": 0.85, "inference_time_ms": 10})
        # Medium confidence
        display_result({"prediction": "ai_generated", "probability": 0.7, "inference_time_ms": 10})
        # Low confidence
        display_result({"prediction": "ai_generated", "probability": 0.55, "inference_time_ms": 10})


class TestModuleConstants:
    """Tests for module-level configuration."""

    def test_constants_defined(self) -> None:
        from src.ui.app import API_URL, MAX_DISPLAY_SIZE, MAX_UPLOAD_MB

        assert isinstance(API_URL, str)
        assert MAX_DISPLAY_SIZE == 800
        assert MAX_UPLOAD_MB == 20
