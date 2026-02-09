"""Input validation utilities for API security."""

import hashlib
import io
import re

from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Magic bytes for image formats
IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # WebP starts with RIFF, need to check for WEBP
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}

# Minimum and maximum dimensions
MIN_IMAGE_SIZE = 10
MAX_IMAGE_SIZE = 10000

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, error_type: str, detail: str) -> None:
        self.error_type = error_type
        self.detail = detail
        super().__init__(f"{error_type}: {detail}")


def detect_mime_type(data: bytes) -> str | None:
    """Detect MIME type from file magic bytes.

    Args:
        data: File content bytes.

    Returns:
        MIME type string or None if not recognized.
    """
    if len(data) < 12:
        return None

    # Check PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    # Check JPEG
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    # Check WebP (RIFF....WEBP)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    # Check GIF
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"

    return None


def validate_image_bytes(
    data: bytes,
    allowed_types: set[str] | None = None,
    max_size: int = MAX_FILE_SIZE,
    min_dimension: int = MIN_IMAGE_SIZE,
    max_dimension: int = MAX_IMAGE_SIZE,
) -> dict:
    """Validate image data thoroughly.

    Args:
        data: Raw image bytes.
        allowed_types: Set of allowed MIME types.
        max_size: Maximum file size in bytes.
        min_dimension: Minimum image dimension.
        max_dimension: Maximum image dimension.

    Returns:
        Dictionary with image metadata.

    Raises:
        ValidationError: If validation fails.
    """
    if allowed_types is None:
        allowed_types = {"image/jpeg", "image/png", "image/webp"}

    # Check file size
    if len(data) > max_size:
        raise ValidationError(
            "file_too_large",
            f"File size {len(data) / (1024 * 1024):.1f}MB exceeds maximum {max_size / (1024 * 1024):.0f}MB",
        )

    if len(data) == 0:
        raise ValidationError("empty_file", "File is empty")

    # Detect actual MIME type from magic bytes
    actual_mime = detect_mime_type(data)
    if actual_mime is None:
        raise ValidationError(
            "invalid_format",
            "Could not detect image format. File may be corrupted or not an image.",
        )

    if actual_mime not in allowed_types:
        raise ValidationError(
            "unsupported_format",
            f"Format {actual_mime} not supported. Allowed: {', '.join(allowed_types)}",
        )

    # Try to open and validate with PIL
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()  # Verify it's a valid image

        # Re-open after verify (verify closes the file)
        image = Image.open(io.BytesIO(data))
        width, height = image.size

    except Exception as e:
        raise ValidationError(
            "corrupt_image",
            f"Image appears to be corrupted: {e}",
        ) from e

    # Validate dimensions
    if width < min_dimension or height < min_dimension:
        raise ValidationError(
            "image_too_small",
            f"Image dimensions {width}x{height} below minimum {min_dimension}x{min_dimension}",
        )

    if width > max_dimension or height > max_dimension:
        raise ValidationError(
            "image_too_large",
            f"Image dimensions {width}x{height} exceed maximum {max_dimension}x{max_dimension}",
        )

    # Calculate content hash for deduplication/logging
    content_hash = hashlib.md5(data).hexdigest()

    return {
        "mime_type": actual_mime,
        "size_bytes": len(data),
        "width": width,
        "height": height,
        "mode": image.mode,
        "content_hash": content_hash,
    }


def validate_content_type(
    claimed_type: str | None,
    actual_type: str,
) -> bool:
    """Validate that claimed content type matches actual.

    Args:
        claimed_type: Content-Type header value.
        actual_type: Detected MIME type.

    Returns:
        True if types match or claimed is None.
    """
    if claimed_type is None:
        return True

    # Normalize types
    claimed = claimed_type.lower().split(";")[0].strip()
    actual = actual_type.lower()

    return claimed == actual


def sanitize_filename(filename: str | None) -> str:
    """Sanitize filename to prevent path traversal.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename.
    """
    if filename is None:
        return "unknown"

    # Replace path traversal patterns (../ or ..\) — dots become underscore
    sanitized = re.sub(r"\.\.[/\\]", lambda m: "_" + m.group()[-1], filename)

    # Remove leading dots (hidden files)
    while sanitized.startswith("."):
        sanitized = sanitized[1:]

    # Remove path separators and null bytes
    sanitized = sanitized.replace("/", "_").replace("\\", "_").replace("\x00", "")

    # Replace any remaining path traversal dots
    sanitized = sanitized.replace("..", "_")

    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    return sanitized or "unknown"


async def validate_upload_file(
    file,
    allowed_types: set[str] | None = None,
    max_size: int = MAX_FILE_SIZE,
) -> tuple[bytes, dict]:
    """Validate an uploaded file.

    Args:
        file: FastAPI UploadFile object.
        allowed_types: Set of allowed MIME types.
        max_size: Maximum file size in bytes.

    Returns:
        Tuple of (file bytes, validation metadata).

    Raises:
        ValidationError: If validation fails.
    """
    # Read file content
    contents = await file.read()

    # Validate image
    metadata = validate_image_bytes(
        contents,
        allowed_types=allowed_types,
        max_size=max_size,
    )

    # Check content type matches
    if not validate_content_type(file.content_type, metadata["mime_type"]):
        logger.warning(
            "Content-Type mismatch",
            claimed=file.content_type,
            actual=metadata["mime_type"],
        )

    # Sanitize filename
    metadata["filename"] = sanitize_filename(file.filename)

    return contents, metadata
