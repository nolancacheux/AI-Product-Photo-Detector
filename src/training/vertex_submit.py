"""Submit training jobs to Vertex AI.

Handles the full lifecycle: upload data to GCS, build and push the Docker
image to Artifact Registry, submit a CustomContainerTrainingJob, and
optionally download the trained model after completion.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from google.cloud import aiplatform

from src.training.gcs import download_file, upload_directory
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# GCP constants
PROJECT_ID = "ai-product-detector-487013"
REGION = "europe-west1"
GCS_BUCKET = "ai-product-detector-487013"
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ai-product-detector"
IMAGE_NAME = "ai-product-detector-train"
IMAGE_TAG = "latest"
IMAGE_URI = f"{ARTIFACT_REGISTRY}/{IMAGE_NAME}:{IMAGE_TAG}"


def build_and_push_image(project_root: Path) -> str:
    """Build the training Docker image and push to Artifact Registry.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Full image URI.
    """
    logger.info("Building Docker image", image=IMAGE_URI)

    # Configure Docker for Artifact Registry
    subprocess.run(
        ["gcloud", "auth", "configure-docker", f"{REGION}-docker.pkg.dev", "--quiet"],
        check=True,
        cwd=str(project_root),
    )

    # Build the image
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            "docker/Dockerfile.training",
            "-t",
            IMAGE_URI,
            ".",
        ],
        check=True,
        cwd=str(project_root),
    )

    # Push to Artifact Registry
    logger.info("Pushing Docker image", image=IMAGE_URI)
    subprocess.run(
        ["docker", "push", IMAGE_URI],
        check=True,
        cwd=str(project_root),
    )

    logger.info("Image pushed successfully", image=IMAGE_URI)
    return IMAGE_URI


def submit_training_job(
    epochs: int,
    batch_size: int,
    config_path: str,
    sync: bool,
) -> aiplatform.CustomContainerTrainingJob:
    """Submit a training job to Vertex AI.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        config_path: Path to training config inside the container.
        sync: Whether to wait for job completion.

    Returns:
        The Vertex AI training job object.
    """
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{GCS_BUCKET}",
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name="ai-product-detector-training",
        container_uri=IMAGE_URI,
    )

    logger.info(
        "Submitting Vertex AI training job",
        machine_type="n1-standard-4",
        accelerator="NVIDIA_TESLA_T4",
        epochs=epochs,
        batch_size=batch_size,
        sync=sync,
    )

    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        boot_disk_size_gb=100,
        args=[
            "--config",
            config_path,
            "--gcs-bucket",
            GCS_BUCKET,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ],
        sync=sync,
    )

    return job


def main() -> None:
    """CLI entry point for Vertex AI training submission."""
    parser = argparse.ArgumentParser(
        description="Submit AI Product Photo Detector training to Vertex AI",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config file inside the container",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="Wait for job completion and download model",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=False,
        help="Skip uploading training data to GCS",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        default=False,
        help="Skip building and pushing the Docker image",
    )
    args = parser.parse_args()

    setup_logging(level="INFO", json_format=False)
    project_root = Path(__file__).resolve().parents[2]

    # Step 1: Upload training data to GCS
    if not args.skip_upload:
        data_dir = project_root / "data" / "processed"
        if not data_dir.exists():
            logger.error("Data directory not found", path=str(data_dir))
            sys.exit(1)

        logger.info("Uploading training data to GCS")
        uploaded = upload_directory(
            local_dir=str(data_dir),
            bucket_name=GCS_BUCKET,
            gcs_prefix="data/processed",
        )
        logger.info("Data upload complete", files_uploaded=uploaded)
    else:
        logger.info("Skipping data upload")

    # Step 2: Build and push Docker image
    if not args.skip_build:
        build_and_push_image(project_root)
    else:
        logger.info("Skipping image build", image=IMAGE_URI)

    # Step 3: Submit training job
    job = submit_training_job(
        epochs=args.epochs,
        batch_size=args.batch_size,
        config_path=args.config,
        sync=args.sync,
    )

    # Step 4: Download trained model (only in sync mode)
    if args.sync:
        logger.info("Downloading trained model from GCS")
        local_model_path = str(project_root / "models" / "checkpoints" / "best_model.pt")
        try:
            download_file(
                bucket_name=GCS_BUCKET,
                gcs_path="models/best_model.pt",
                local_path=local_model_path,
            )
            logger.info("Model saved", path=local_model_path)
        except Exception:
            logger.exception("Failed to download model from GCS")
            sys.exit(1)
    else:
        logger.info(
            "Job submitted (async). Use --sync to wait and download the model.",
            job_name=job.display_name,
        )


if __name__ == "__main__":
    main()
