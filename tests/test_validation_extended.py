"""Extended tests for input validation (src/inference/validation.py)."""

import io
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.inference.validation import (
    ValidationError,
    detect_mime_type,
    sanitize_filename,
    validate_image_bytes,
    validate_upload_file,
)


def _make_webp_bytes(size: tuple[int, int] = (100, 100)) -> bytes:
    """Create valid WebP image bytes."""
    img = Image.new("RGB", size, "green")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _make_gif_bytes(size: tuple[int, int] = (100, 100)) -> bytes:
    """Create valid GIF image bytes."""
    img = Image.new("P", size)
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return buf.getvalue()


class TestDetectMimeTypeExtended:
    """Extended tests for MIME type detection."""

    def test_detect_webp(self) -> None:
        data = _make_webp_bytes()
        assert detect_mime_type(data) == "image/webp"

    def test_detect_gif87a(self) -> None:
        data = b"GIF87a" + b"\x00" * 100
        assert detect_mime_type(data) == "image/gif"

    def test_detect_gif89a(self) -> None:
        data = _make_gif_bytes()
        assert detect_mime_type(data) == "image/gif"

    def test_empty_data(self) -> None:
        assert detect_mime_type(b"") is None

    def test_exactly_12_bytes(self) -> None:
        # 12 bytes but no magic match
        assert detect_mime_type(b"123456789012") is None


class TestValidateImageBytesExtended:
    """Extended tests for validate_image_bytes()."""

    def test_webp_image(self) -> None:
        data = _make_webp_bytes()
        result = validate_image_bytes(data)
        assert result["mime_type"] == "image/webp"
        assert result["width"] == 100
        assert result["height"] == 100

    def test_gif_not_in_default_allowed(self) -> None:
        data = _make_gif_bytes()
        with pytest.raises(ValidationError) as exc_info:
            validate_image_bytes(data)
        assert exc_info.value.error_type == "unsupported_format"

    def test_gif_in_custom_allowed(self) -> None:
        data = _make_gif_bytes()
        result = validate_image_bytes(data, allowed_types={"image/gif", "image/jpeg", "image/png"})
        assert result["mime_type"] == "image/gif"

    def test_image_too_large_dimension(self) -> None:
        img = Image.new("RGB", (200, 200), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        with pytest.raises(ValidationError) as exc_info:
            validate_image_bytes(data, max_dimension=100)
        assert exc_info.value.error_type == "image_too_large"

    def test_content_hash_is_sha256(self) -> None:
        img = Image.new("RGB", (50, 50), "blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        result = validate_image_bytes(data, min_dimension=1)
        assert len(result["content_hash"]) == 64  # SHA-256 hex length


class TestSanitizeFilenameExtended:
    """Extended edge case tests for sanitize_filename()."""

    def test_backslash_traversal(self) -> None:
        result = sanitize_filename("..\\..\\etc\\passwd")
        assert ".." not in result
        assert "\\" not in result

    def test_empty_string(self) -> None:
        assert sanitize_filename("") == "unknown"

    def test_only_dots(self) -> None:
        assert sanitize_filename("...") == "unknown"

    def test_slash_replacement(self) -> None:
        result = sanitize_filename("path/to/file.jpg")
        assert "/" not in result


class TestValidateUploadFile:
    """Tests for validate_upload_file()."""

    @pytest.mark.asyncio
    async def test_valid_upload(self) -> None:
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        mock_file = AsyncMock()
        mock_file.read.return_value = data
        mock_file.content_type = "image/jpeg"
        mock_file.filename = "test.jpg"

        contents, metadata = await validate_upload_file(mock_file)
        assert contents == data
        assert metadata["mime_type"] == "image/jpeg"
        assert metadata["filename"] == "test.jpg"

    @pytest.mark.asyncio
    async def test_content_type_mismatch_logged(self) -> None:
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        mock_file = AsyncMock()
        mock_file.read.return_value = data
        mock_file.content_type = "image/png"  # Mismatch
        mock_file.filename = "test.jpg"

        contents, metadata = await validate_upload_file(mock_file)
        # Should still succeed but log warning
        assert metadata["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_malicious_filename_sanitized(self) -> None:
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        mock_file = AsyncMock()
        mock_file.read.return_value = data
        mock_file.content_type = "image/jpeg"
        mock_file.filename = "../../../etc/passwd"

        _, metadata = await validate_upload_file(mock_file)
        assert ".." not in metadata["filename"]
        assert "/" not in metadata["filename"]
