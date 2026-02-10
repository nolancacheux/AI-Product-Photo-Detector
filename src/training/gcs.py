"""Google Cloud Storage helpers for training pipelines.

Provides upload/download utilities used by both the local training script
and the Vertex AI submission script. All functions are no-ops friendly:
callers gate on whether a GCS bucket was provided.
"""

from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROJECT = "ai-product-detector-487013"


def _get_client(project: str = DEFAULT_PROJECT) -> Any:
    """Get a Google Cloud Storage client (lazy import).

    Args:
        project: GCP project ID.

    Returns:
        google.cloud.storage.Client instance.
    """
    from google.cloud import storage
    return storage.Client(project=project)


def download_directory(
    bucket_name: str,
    gcs_prefix: str,
    local_dir: str,
) -> int:
    """Download a directory from GCS to a local path.

    Args:
        bucket_name: GCS bucket name (without gs:// prefix).
        gcs_prefix: Prefix (folder) in the bucket to download.
        local_dir: Local directory to download into.

    Returns:
        Number of files downloaded.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    count = 0

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = blob.name[len(gcs_prefix):].lstrip("/")
        if not relative_path:
            continue

        local_path = Path(local_dir) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        count += 1

    logger.info(
        "Downloaded from GCS",
        bucket=bucket_name,
        prefix=gcs_prefix,
        files=count,
        local_dir=local_dir,
    )
    return count


def upload_file(
    local_path: str,
    bucket_name: str,
    gcs_path: str,
) -> None:
    """Upload a single file to GCS.

    Args:
        local_path: Path to the local file.
        bucket_name: GCS bucket name.
        gcs_path: Destination path in the bucket.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logger.info(
        "Uploaded to GCS",
        source=local_path,
        destination=f"gs://{bucket_name}/{gcs_path}",
    )


def upload_directory(
    local_dir: str,
    bucket_name: str,
    gcs_prefix: str,
) -> int:
    """Upload a local directory recursively to GCS.

    Skips files that already exist in GCS with the same size.

    Args:
        local_dir: Local directory path.
        bucket_name: GCS bucket name.
        gcs_prefix: Destination prefix in the bucket.

    Returns:
        Number of files uploaded.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)
    count = 0

    for file_path in local_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(local_path)
        blob_name = f"{gcs_prefix}/{relative}"
        blob = bucket.blob(blob_name)

        # Skip if already uploaded and same size
        if blob.exists():
            blob.reload()
            if blob.size == file_path.stat().st_size:
                continue

        blob.upload_from_filename(str(file_path))
        count += 1

    logger.info(
        "Uploaded directory to GCS",
        local_dir=local_dir,
        destination=f"gs://{bucket_name}/{gcs_prefix}",
        files=count,
    )
    return count


def download_file(
    bucket_name: str,
    gcs_path: str,
    local_path: str,
) -> None:
    """Download a single file from GCS.

    Args:
        bucket_name: GCS bucket name.
        gcs_path: Path within the bucket.
        local_path: Local destination path.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    local_file = Path(local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(str(local_file))
    logger.info(
        "Downloaded from GCS",
        source=f"gs://{bucket_name}/{gcs_path}",
        destination=local_path,
    )
