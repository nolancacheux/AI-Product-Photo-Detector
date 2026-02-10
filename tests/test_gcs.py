"""Tests for the GCS helper module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.gcs import (
    download_directory,
    download_file,
    upload_directory,
    upload_file,
)


@pytest.fixture()
def mock_gcs_client():
    """Provide a mocked GCS client with bucket and blob helpers."""
    with patch("src.training.gcs._get_client") as mock_get:
        client = MagicMock()
        bucket = MagicMock()
        client.bucket.return_value = bucket
        mock_get.return_value = client
        yield client, bucket


class TestUploadFile:
    """Tests for upload_file()."""

    def test_uploads_single_file(self, mock_gcs_client: tuple) -> None:
        client, bucket = mock_gcs_client
        blob = MagicMock()
        bucket.blob.return_value = blob

        upload_file("/tmp/model.pt", "my-bucket", "models/model.pt")

        client.bucket.assert_called_once_with("my-bucket")
        bucket.blob.assert_called_once_with("models/model.pt")
        blob.upload_from_filename.assert_called_once_with("/tmp/model.pt")


class TestDownloadFile:
    """Tests for download_file()."""

    def test_downloads_single_file(self, mock_gcs_client: tuple, tmp_path: Path) -> None:
        client, bucket = mock_gcs_client
        blob = MagicMock()
        bucket.blob.return_value = blob

        dest = str(tmp_path / "sub" / "model.pt")
        download_file("my-bucket", "models/model.pt", dest)

        client.bucket.assert_called_once_with("my-bucket")
        bucket.blob.assert_called_once_with("models/model.pt")
        blob.download_to_filename.assert_called_once_with(dest)


class TestUploadDirectory:
    """Tests for upload_directory()."""

    def test_uploads_files_recursively(self, mock_gcs_client: tuple, tmp_path: Path) -> None:
        _, bucket = mock_gcs_client

        # Create a small directory tree
        (tmp_path / "a.txt").write_text("a")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")

        blob_mock = MagicMock()
        blob_mock.exists.return_value = False
        bucket.blob.return_value = blob_mock

        count = upload_directory(str(tmp_path), "my-bucket", "prefix")

        assert count == 2
        assert blob_mock.upload_from_filename.call_count == 2

    def test_skips_existing_blobs_with_same_size(
        self,
        mock_gcs_client: tuple,
        tmp_path: Path,
    ) -> None:
        _, bucket = mock_gcs_client

        content = "hello"
        (tmp_path / "file.txt").write_text(content)
        file_size = (tmp_path / "file.txt").stat().st_size

        blob_mock = MagicMock()
        blob_mock.exists.return_value = True
        blob_mock.size = file_size
        bucket.blob.return_value = blob_mock

        count = upload_directory(str(tmp_path), "my-bucket", "prefix")

        assert count == 0
        blob_mock.upload_from_filename.assert_not_called()

    def test_reuploads_when_size_differs(
        self,
        mock_gcs_client: tuple,
        tmp_path: Path,
    ) -> None:
        _, bucket = mock_gcs_client

        (tmp_path / "file.txt").write_text("hello")

        blob_mock = MagicMock()
        blob_mock.exists.return_value = True
        blob_mock.size = 999  # different from actual file size
        bucket.blob.return_value = blob_mock

        count = upload_directory(str(tmp_path), "my-bucket", "prefix")

        assert count == 1
        blob_mock.upload_from_filename.assert_called_once()

    def test_empty_directory_uploads_nothing(
        self,
        mock_gcs_client: tuple,
        tmp_path: Path,
    ) -> None:
        count = upload_directory(str(tmp_path), "my-bucket", "prefix")
        assert count == 0


class TestDownloadDirectory:
    """Tests for download_directory()."""

    def test_downloads_blobs(self, mock_gcs_client: tuple, tmp_path: Path) -> None:
        _, bucket = mock_gcs_client

        blob1 = MagicMock()
        blob1.name = "data/processed/train/real/img1.jpg"
        blob2 = MagicMock()
        blob2.name = "data/processed/train/ai_generated/img2.jpg"
        # Directory-like blob should be skipped
        dir_blob = MagicMock()
        dir_blob.name = "data/processed/train/"

        bucket.list_blobs.return_value = [dir_blob, blob1, blob2]

        count = download_directory("my-bucket", "data/processed", str(tmp_path))

        assert count == 2
        blob1.download_to_filename.assert_called_once()
        blob2.download_to_filename.assert_called_once()
        dir_blob.download_to_filename.assert_not_called()

    def test_skips_empty_relative_path(self, mock_gcs_client: tuple, tmp_path: Path) -> None:
        _, bucket = mock_gcs_client

        # Blob whose name equals the prefix exactly (relative path empty after strip)
        blob = MagicMock()
        blob.name = "data/processed"
        bucket.list_blobs.return_value = [blob]

        count = download_directory("my-bucket", "data/processed", str(tmp_path))
        assert count == 0

    def test_returns_zero_for_empty_bucket(
        self,
        mock_gcs_client: tuple,
        tmp_path: Path,
    ) -> None:
        _, bucket = mock_gcs_client
        bucket.list_blobs.return_value = []

        count = download_directory("my-bucket", "data/processed", str(tmp_path))
        assert count == 0
