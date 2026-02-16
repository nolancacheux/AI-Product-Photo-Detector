"""Shared GCP constants for the AI Product Photo Detector project.

All values read from environment variables with hardcoded fallback defaults.
Import these instead of redefining them in each module.
"""

import os

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ai-product-detector-487013")
REGION = os.getenv("GCP_REGION", "europe-west1")
GCS_BUCKET = os.getenv("GCP_GCS_BUCKET", "ai-product-detector-487013")
GCS_BUCKET_URI = f"gs://{GCS_BUCKET}"
PIPELINE_ROOT = f"{GCS_BUCKET_URI}/pipeline_root"
ARTIFACT_REGISTRY = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ai-product-detector"
)
TRAINING_IMAGE = f"{ARTIFACT_REGISTRY}/train:latest"
SERVING_IMAGE = f"{ARTIFACT_REGISTRY}/serve:latest"
SERVICE_NAME = os.getenv("GCP_SERVICE_NAME", "ai-product-detector")
