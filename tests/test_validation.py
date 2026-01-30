"""Tests for input validation module."""

import io

import pytest
from PIL import Image

from src.inference.validation import (
    ValidationError,
    detect_mime_type,
    sanitize_filename,
    validate_content_type,
    validate_image_bytes,
)


@pytest.fixture
def jpeg_bytes() -> bytes:
    """Create valid JPEG bytes."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def png_bytes() -> bytes:
    """Create valid PNG bytes."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


class TestDetectMimeType:
    """Tests for MIME type detection."""

    def test_detect_jpeg(self, jpeg_bytes: bytes) -> None:
        """Test JPEG detection."""
        assert detect_mime_type(jpeg_bytes) == "image/jpeg"

    def test_detect_png(self, png_bytes: bytes) -> None:
        """Test PNG detection."""
        assert detect_mime_type(png_bytes) == "image/png"

    def test_detect_unknown(self) -> None:
        """Test unknown format returns None."""
        assert detect_mime_type(b"random bytes") is None

    def test_detect_short_data(self) -> None:
        """Test short data returns None."""
        assert detect_mime_type(b"abc") is None


class TestValidateImageBytes:
    """Tests for image validation."""

    def test_valid_jpeg(self, jpeg_bytes: bytes) -> None:
        """Test valid JPEG passes validation."""
        result = validate_image_bytes(jpeg_bytes)
        assert result["mime_type"] == "image/jpeg"
        assert result["width"] == 100
        assert result["height"] == 100
        assert "content_hash" in result

    def test_valid_png(self, png_bytes: bytes) -> None:
        """Test valid PNG passes validation."""
        result = validate_image_bytes(png_bytes)
        assert result["mime_type"] == "image/png"

    def test_empty_file_rejected(self) -> None:
        """Test empty file is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_image_bytes(b"")
        assert exc_info.value.error_type == "empty_file"

    def test_unsupported_format_rejected(self) -> None:
        """Test unsupported format is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_image_bytes(b"random data that is not an image")
        assert exc_info.value.error_type == "invalid_format"

    def test_file_too_large_rejected(self, jpeg_bytes: bytes) -> None:
        """Test oversized file is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_image_bytes(jpeg_bytes, max_size=100)
        assert exc_info.value.error_type == "file_too_large"

    def test_image_too_small_rejected(self) -> None:
        """Test small image is rejected."""
        img = Image.new("RGB", (5, 5), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        with pytest.raises(ValidationError) as exc_info:
            validate_image_bytes(buffer.read(), min_dimension=10)
        assert exc_info.value.error_type == "image_too_small"


class TestValidateContentType:
    """Tests for content type validation."""

    def test_matching_types(self) -> None:
        """Test matching types pass."""
        assert validate_content_type("image/jpeg", "image/jpeg") is True

    def test_none_claimed_type(self) -> None:
        """Test None claimed type passes."""
        assert validate_content_type(None, "image/jpeg") is True

    def test_mismatched_types(self) -> None:
        """Test mismatched types fail."""
        assert validate_content_type("image/png", "image/jpeg") is False

    def test_content_type_with_params(self) -> None:
        """Test content type with parameters."""
        assert validate_content_type("image/jpeg; charset=utf-8", "image/jpeg") is True


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_normal_filename(self) -> None:
        """Test normal filename unchanged."""
        assert sanitize_filename("test.jpg") == "test.jpg"

    def test_path_traversal_removed(self) -> None:
        """Test path traversal is removed."""
        assert sanitize_filename("../../../etc/passwd") == "______etc_passwd"

    def test_null_bytes_removed(self) -> None:
        """Test null bytes are removed."""
        assert sanitize_filename("test\x00.jpg") == "test.jpg"

    def test_hidden_files_exposed(self) -> None:
        """Test hidden file dots are removed."""
        assert sanitize_filename("...hidden") == "hidden"

    def test_none_returns_unknown(self) -> None:
        """Test None returns 'unknown'."""
        assert sanitize_filename(None) == "unknown"

    def test_long_filename_truncated(self) -> None:
        """Test long filename is truncated."""
        long_name = "a" * 300 + ".jpg"
        result = sanitize_filename(long_name)
        assert len(result) <= 255
