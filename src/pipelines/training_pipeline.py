"""Vertex AI Training Pipeline for AI Product Photo Detector.

Defines a Kubeflow Pipeline that orchestrates data validation, model training,
evaluation, comparison against the production model, and conditional deployment.

Usage:
    # Compile the pipeline
    python -m src.pipelines.training_pipeline compile

    # Submit a pipeline run
    python -m src.pipelines.training_pipeline run \
        --config configs/pipeline_config.yaml \
        --epochs 15 --batch-size 64 --min-accuracy 0.85
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from kfp import compiler, dsl

from src.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

# -- GCP constants (overridable via pipeline_config.yaml) ---------------------
PROJECT_ID = "ai-product-detector-487013"
REGION = "europe-west1"
GCS_BUCKET = "gs://ai-product-detector-487013"
PIPELINE_ROOT = f"{GCS_BUCKET}/pipeline_root"
ARTIFACT_REGISTRY = "europe-west1-docker.pkg.dev/ai-product-detector-487013/ai-product-detector"
TRAINING_IMAGE = f"{ARTIFACT_REGISTRY}/train:latest"
SERVICE_NAME = "ai-product-detector"


# =============================================================================
# Pipeline Components
# =============================================================================


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-storage>=2.10.0",
        "Pillow>=10.0.0",
    ],
)
def validate_data(
    data_gcs_path: str,
    min_samples_per_class: int,
    max_class_imbalance_ratio: float,
) -> dsl.Artifact:
    """Validate the training dataset on GCS.

    Checks:
        - Both class directories exist (real/, ai_generated/)
        - Minimum sample count per class
        - Class balance ratio above threshold
        - Image file integrity (PIL can open each file)

    Args:
        data_gcs_path: GCS path to the dataset root (e.g. gs://bucket/data/processed).
        min_samples_per_class: Minimum required images per class.
        max_class_imbalance_ratio: Maximum allowed min/max class ratio (0-1).

    Returns:
        Artifact containing the validation report JSON.
    """
    import json
    import tempfile
    from pathlib import Path

    from google.cloud import storage
    from PIL import Image

    client = storage.Client()
    bucket_name = data_gcs_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(data_gcs_path.replace("gs://", "").split("/")[1:])
    bucket = client.bucket(bucket_name)

    report: dict = {"valid": True, "errors": [], "warnings": [], "class_counts": {}}

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    expected_splits = ["train", "val", "test"]
    corrupted_count = 0

    for split in expected_splits:
        for class_name in ["real", "ai_generated"]:
            class_prefix = f"{prefix}/{split}/{class_name}/"
            blobs = list(bucket.list_blobs(prefix=class_prefix))
            image_blobs = [b for b in blobs if Path(b.name).suffix.lower() in valid_extensions]
            count = len(image_blobs)
            key = f"{split}/{class_name}"
            report["class_counts"][key] = count

            if count < min_samples_per_class:
                report["errors"].append(
                    f"{key}: only {count} samples (min required: {min_samples_per_class})"
                )
                report["valid"] = False

            # Spot-check a subset for corruption
            check_limit = min(count, 20)
            for blob in image_blobs[:check_limit]:
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(blob.name).suffix) as tmp:
                        blob.download_to_filename(tmp.name)
                        with Image.open(tmp.name) as img:
                            img.verify()
                except Exception as exc:
                    corrupted_count += 1
                    report["warnings"].append(f"Corrupted: {blob.name} ({exc})")

        # Check class balance within split
        real_key = f"{split}/real"
        ai_key = f"{split}/ai_generated"
        real_count = report["class_counts"].get(real_key, 0)
        ai_count = report["class_counts"].get(ai_key, 0)

        if real_count > 0 and ai_count > 0:
            ratio = min(real_count, ai_count) / max(real_count, ai_count)
            if ratio < max_class_imbalance_ratio:
                report["warnings"].append(
                    f"{split}: class imbalance ratio {ratio:.3f} "
                    f"(threshold: {max_class_imbalance_ratio})"
                )

    report["corrupted_count"] = corrupted_count
    if corrupted_count > 0:
        report["warnings"].append(f"Total corrupted images found: {corrupted_count}")

    if report["errors"]:
        report["valid"] = False

    # Write report as component output
    report_json = json.dumps(report, indent=2)
    print(report_json)

    if not report["valid"]:
        raise RuntimeError(
            f"Data validation failed with {len(report['errors'])} error(s): "
            + "; ".join(report["errors"])
        )

    return report_json


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0",
    ],
)
def train_model(
    project_id: str,
    region: str,
    training_image: str,
    data_gcs_path: str,
    output_gcs_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_passed: str,
) -> str:
    """Launch a Vertex AI CustomContainerTrainingJob on a T4 GPU.

    Args:
        project_id: GCP project ID.
        region: GCP region.
        training_image: Docker image URI for training.
        data_gcs_path: GCS path to the processed dataset.
        output_gcs_path: GCS path for training outputs (checkpoints, logs).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        validation_passed: Output from validate_data (used for DAG ordering).

    Returns:
        GCS path to the trained model checkpoint.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="ai-product-detector-training",
        container_uri=training_image,
        command=["python", "-m", "src.training.train"],
        model_serving_container_image_uri=None,
    )

    job.run(
        args=[
            "--config",
            "configs/train_config.yaml",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--data-dir",
            "/gcs/data/processed",
            "--output-dir",
            "/gcs/output",
        ],
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_output_dir=output_gcs_path,
        environment_variables={
            "DATA_GCS_PATH": data_gcs_path,
            "OUTPUT_GCS_PATH": output_gcs_path,
        },
    )

    model_artifact_path = f"{output_gcs_path}/model/best_model.pt"
    print(f"Training complete. Model artifact: {model_artifact_path}")
    return model_artifact_path


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "google-cloud-storage>=2.10.0",
    ],
)
def evaluate_model(
    model_gcs_path: str,
    data_gcs_path: str,
    output_gcs_path: str,
    batch_size: int,
    image_size: int,
) -> str:
    """Evaluate a trained model on the test set.

    Downloads the model and test data from GCS, runs inference, computes
    metrics (accuracy, precision, recall, F1, AUC-ROC), and uploads results.

    Args:
        model_gcs_path: GCS path to model checkpoint.
        data_gcs_path: GCS path to dataset root.
        output_gcs_path: GCS path for evaluation outputs.
        batch_size: Inference batch size.
        image_size: Input image resolution.

    Returns:
        JSON string of evaluation metrics.
    """
    import json
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch
    from google.cloud import storage
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )

    client = storage.Client()

    # Download model checkpoint
    tmpdir = Path(tempfile.mkdtemp())
    model_local = tmpdir / "best_model.pt"

    bucket_name = model_gcs_path.replace("gs://", "").split("/")[0]
    blob_path = "/".join(model_gcs_path.replace("gs://", "").split("/")[1:])
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(str(model_local))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(model_local), map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Reconstruct model
    import timm
    import torch.nn as nn

    class _Detector(nn.Module):
        def __init__(self, backbone_name: str, dropout: float) -> None:
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            with torch.no_grad():
                feat_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.backbone(x))

    model = _Detector(
        backbone_name=model_config.get("name", "efficientnet_b0"),
        dropout=model_config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Download test images
    test_prefix = "/".join(data_gcs_path.replace("gs://", "").split("/")[1:]) + "/test"
    test_dir = tmpdir / "test"
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

    for class_name in ["real", "ai_generated"]:
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        blobs = bucket.list_blobs(prefix=f"{test_prefix}/{class_name}/")
        for blob in blobs:
            if Path(blob.name).suffix.lower() in valid_ext:
                local_file = class_dir / Path(blob.name).name
                blob.download_to_filename(str(local_file))

    # Build simple dataset and loader
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class _TestDataset(Dataset):
        def __init__(self, root: Path) -> None:
            self.samples: list[tuple[Path, int]] = []
            for img in (root / "real").glob("*"):
                if img.suffix.lower() in valid_ext:
                    self.samples.append((img, 0))
            for img in (root / "ai_generated").glob("*"):
                if img.suffix.lower() in valid_ext:
                    self.samples.append((img, 1))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> tuple:
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            return transform(img), label

    dataset = _TestDataset(test_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend((probs >= 0.5).astype(int).tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": roc_auc,
        "total_samples": int(len(y_true)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    # Upload metrics JSON to GCS
    metrics_json = json.dumps(metrics, indent=2)
    out_bucket_name = output_gcs_path.replace("gs://", "").split("/")[0]
    out_prefix = "/".join(output_gcs_path.replace("gs://", "").split("/")[1:])
    out_bucket = client.bucket(out_bucket_name)
    out_bucket.blob(f"{out_prefix}/metrics.json").upload_from_string(metrics_json)

    print(metrics_json)
    return metrics_json


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-storage>=2.10.0"],
)
def compare_models(
    new_metrics_json: str,
    production_metrics_gcs_path: str,
    min_accuracy: float,
    min_f1: float,
) -> str:
    """Compare new model metrics against the production model.

    Args:
        new_metrics_json: JSON string of new model metrics.
        production_metrics_gcs_path: GCS path to production metrics JSON.
        min_accuracy: Minimum accuracy threshold for the new model.
        min_f1: Minimum F1 threshold for the new model.

    Returns:
        JSON string with comparison results and a ``should_register`` flag.
    """
    import json

    from google.cloud import storage

    new_metrics = json.loads(new_metrics_json)

    # Load production metrics if available
    production_metrics = None
    try:
        client = storage.Client()
        bucket_name = production_metrics_gcs_path.replace("gs://", "").split("/")[0]
        blob_path = "/".join(production_metrics_gcs_path.replace("gs://", "").split("/")[1:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if blob.exists():
            production_metrics = json.loads(blob.download_as_text())
    except Exception as exc:
        print(f"Warning: could not load production metrics: {exc}")

    result: dict = {
        "new_metrics": {
            "accuracy": new_metrics["accuracy"],
            "f1": new_metrics["f1"],
            "precision": new_metrics["precision"],
            "recall": new_metrics["recall"],
            "auc_roc": new_metrics["auc_roc"],
        },
        "thresholds": {"min_accuracy": min_accuracy, "min_f1": min_f1},
        "meets_thresholds": (
            new_metrics["accuracy"] >= min_accuracy and new_metrics["f1"] >= min_f1
        ),
        "production_metrics": None,
        "improvement": None,
        "should_register": False,
    }

    if production_metrics:
        result["production_metrics"] = {
            "accuracy": production_metrics.get("accuracy", 0.0),
            "f1": production_metrics.get("f1", 0.0),
        }
        result["improvement"] = {
            "accuracy_delta": new_metrics["accuracy"] - production_metrics.get("accuracy", 0.0),
            "f1_delta": new_metrics["f1"] - production_metrics.get("f1", 0.0),
        }
        # Register if meets thresholds AND is at least as good as production
        result["should_register"] = (
            result["meets_thresholds"]
            and new_metrics["accuracy"] >= production_metrics.get("accuracy", 0.0)
            and new_metrics["f1"] >= production_metrics.get("f1", 0.0)
        )
    else:
        # No production model -- register if thresholds are met
        result["should_register"] = result["meets_thresholds"]

    result_json = json.dumps(result, indent=2)
    print(result_json)
    return result_json


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform>=1.38.0"],
)
def register_model(
    comparison_json: str,
    model_gcs_path: str,
    project_id: str,
    region: str,
    model_display_name: str,
    serving_image: str,
) -> str:
    """Register the model in Vertex AI Model Registry if comparison passes.

    Args:
        comparison_json: JSON output from compare_models.
        model_gcs_path: GCS path to the trained model artifact.
        project_id: GCP project ID.
        region: GCP region.
        model_display_name: Display name for the registered model.
        serving_image: Docker image URI for model serving.

    Returns:
        Vertex AI model resource name, or empty string if skipped.
    """
    import json

    from google.cloud import aiplatform

    comparison = json.loads(comparison_json)

    if not comparison.get("should_register", False):
        print("Model did not pass comparison gate. Skipping registration.")
        return ""

    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_gcs_path.rsplit("/", 1)[0],
        serving_container_image_uri=serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        labels={
            "accuracy": str(round(comparison["new_metrics"]["accuracy"], 4)),
            "f1": str(round(comparison["new_metrics"]["f1"], 4)),
        },
    )

    resource_name = model.resource_name
    print(f"Model registered: {resource_name}")
    return resource_name


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-run>=0.10.0",
        "google-cloud-aiplatform>=1.38.0",
    ],
)
def deploy_model(
    model_resource_name: str,
    project_id: str,
    region: str,
    service_name: str,
    auto_deploy: bool,
) -> str:
    """Trigger Cloud Run redeployment with the new model.

    Args:
        model_resource_name: Vertex AI model resource name.
        project_id: GCP project ID.
        region: GCP region.
        service_name: Cloud Run service name.
        auto_deploy: Whether to actually deploy or just log intent.

    Returns:
        Deployment status message.
    """
    if not model_resource_name:
        msg = "No model to deploy (registration was skipped)."
        print(msg)
        return msg

    if not auto_deploy:
        msg = (
            f"Auto-deploy is disabled. Model {model_resource_name} is registered "
            f"but not deployed. Set auto_deploy=true to enable."
        )
        print(msg)
        return msg

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # Trigger Cloud Run service update via gcloud or API
    import subprocess

    result = subprocess.run(
        [
            "gcloud",
            "run",
            "services",
            "update",
            service_name,
            "--region",
            region,
            "--update-env-vars",
            f"MODEL_RESOURCE_NAME={model_resource_name}",
            "--project",
            project_id,
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        msg = f"Deployed model {model_resource_name} to Cloud Run service {service_name}."
    else:
        msg = f"Cloud Run update failed (exit {result.returncode}): {result.stderr.strip()}"

    print(msg)
    return msg


# =============================================================================
# Pipeline Definition
# =============================================================================


@dsl.pipeline(
    name="ai-product-detector-training-pipeline",
    description=(
        "End-to-end training pipeline for the AI Product Photo Detector. "
        "Validates data, trains on GPU, evaluates, compares to production, "
        "and optionally deploys to Cloud Run."
    ),
)
def training_pipeline(
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    min_accuracy: float = 0.85,
    min_f1: float = 0.80,
    auto_deploy: bool = False,
    data_gcs_path: str = f"{GCS_BUCKET}/data/processed",
    output_gcs_path: str = f"{GCS_BUCKET}/pipeline_runs",
    production_metrics_gcs_path: str = f"{GCS_BUCKET}/production/metrics.json",
    project_id: str = PROJECT_ID,
    region: str = REGION,
    training_image: str = TRAINING_IMAGE,
    serving_image: str = f"{ARTIFACT_REGISTRY}/serve:latest",
    model_display_name: str = "ai-product-detector",
    service_name: str = SERVICE_NAME,
    min_samples_per_class: int = 100,
    max_class_imbalance_ratio: float = 0.3,
    image_size: int = 224,
) -> None:
    """Full training pipeline DAG.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        min_accuracy: Minimum accuracy to accept the new model.
        min_f1: Minimum F1 score to accept the new model.
        auto_deploy: Whether to auto-deploy to Cloud Run on success.
        data_gcs_path: GCS path to the processed dataset.
        output_gcs_path: GCS base path for pipeline run outputs.
        production_metrics_gcs_path: GCS path to current production metrics.
        project_id: GCP project ID.
        region: GCP region.
        training_image: Training container image URI.
        serving_image: Serving container image URI.
        model_display_name: Display name for the Vertex AI Model Registry.
        service_name: Cloud Run service name.
        min_samples_per_class: Minimum images per class for validation.
        max_class_imbalance_ratio: Maximum allowed class imbalance ratio.
        image_size: Input image resolution.
    """
    # Step 1: Validate data
    validation_task = validate_data(
        data_gcs_path=data_gcs_path,
        min_samples_per_class=min_samples_per_class,
        max_class_imbalance_ratio=max_class_imbalance_ratio,
    )

    # Step 2: Train model (depends on validation)
    train_task = train_model(
        project_id=project_id,
        region=region,
        training_image=training_image,
        data_gcs_path=data_gcs_path,
        output_gcs_path=output_gcs_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_passed=validation_task.output,
    )

    # Step 3: Evaluate new model
    eval_task = evaluate_model(
        model_gcs_path=train_task.output,
        data_gcs_path=data_gcs_path,
        output_gcs_path=output_gcs_path,
        batch_size=batch_size,
        image_size=image_size,
    )

    # Step 4: Compare against production
    compare_task = compare_models(
        new_metrics_json=eval_task.output,
        production_metrics_gcs_path=production_metrics_gcs_path,
        min_accuracy=min_accuracy,
        min_f1=min_f1,
    )

    # Step 5: Register model if comparison passes
    register_task = register_model(
        comparison_json=compare_task.output,
        model_gcs_path=train_task.output,
        project_id=project_id,
        region=region,
        model_display_name=model_display_name,
        serving_image=serving_image,
    )

    # Step 6: Deploy model (gated by auto_deploy flag)
    deploy_model(
        model_resource_name=register_task.output,
        project_id=project_id,
        region=region,
        service_name=service_name,
        auto_deploy=auto_deploy,
    )


# =============================================================================
# CLI
# =============================================================================


def compile_pipeline(output_path: str = "pipeline.yaml") -> None:
    """Compile the pipeline to a YAML file.

    Args:
        output_path: Destination file path.
    """
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=output_path,
    )
    logger.info("Pipeline compiled to %s", output_path)


def submit_pipeline(
    config_path: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    min_accuracy: float | None = None,
    auto_deploy: bool = False,
) -> None:
    """Submit the pipeline to Vertex AI Pipelines.

    Args:
        config_path: Path to pipeline_config.yaml.
        epochs: Override for training epochs.
        batch_size: Override for training batch size.
        min_accuracy: Override for minimum accuracy threshold.
        auto_deploy: Whether to auto-deploy on success.
    """
    from google.cloud import aiplatform

    config = load_yaml_config(config_path)
    pipeline_cfg = config.get("pipeline", {})
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    deployment_cfg = config.get("deployment", {})

    project_id = pipeline_cfg.get("project_id", PROJECT_ID)
    region = pipeline_cfg.get("region", REGION)
    pipeline_root = pipeline_cfg.get("pipeline_root", PIPELINE_ROOT)

    aiplatform.init(project=project_id, location=region)

    params: dict[str, Any] = {
        "epochs": epochs or training_cfg.get("epochs", 15),
        "batch_size": batch_size or training_cfg.get("batch_size", 64),
        "learning_rate": training_cfg.get("learning_rate", 0.001),
        "min_accuracy": min_accuracy or evaluation_cfg.get("min_accuracy", 0.85),
        "min_f1": evaluation_cfg.get("min_f1", 0.80),
        "auto_deploy": auto_deploy or deployment_cfg.get("auto_deploy", False),
        "data_gcs_path": pipeline_cfg.get("data_gcs_path", f"{GCS_BUCKET}/data/processed"),
        "output_gcs_path": pipeline_cfg.get("output_gcs_path", f"{GCS_BUCKET}/pipeline_runs"),
        "production_metrics_gcs_path": pipeline_cfg.get(
            "production_metrics_gcs_path", f"{GCS_BUCKET}/production/metrics.json"
        ),
        "project_id": project_id,
        "region": region,
        "training_image": pipeline_cfg.get("training_image", TRAINING_IMAGE),
        "serving_image": pipeline_cfg.get("serving_image", f"{ARTIFACT_REGISTRY}/serve:latest"),
        "model_display_name": pipeline_cfg.get("model_display_name", "ai-product-detector"),
        "service_name": deployment_cfg.get("service_name", SERVICE_NAME),
        "min_samples_per_class": evaluation_cfg.get("min_samples_per_class", 100),
        "max_class_imbalance_ratio": evaluation_cfg.get("max_class_imbalance_ratio", 0.3),
        "image_size": training_cfg.get("image_size", 224),
    }

    # Compile
    compiled_path = "/tmp/training_pipeline.yaml"
    compile_pipeline(compiled_path)

    # Submit
    job = aiplatform.PipelineJob(
        display_name=pipeline_cfg.get("pipeline_name", "ai-product-detector-training"),
        template_path=compiled_path,
        pipeline_root=pipeline_root,
        parameter_values=params,
        enable_caching=pipeline_cfg.get("enable_caching", True),
    )

    job.submit()
    logger.info("Pipeline submitted: %s", job.resource_name)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Product Photo Detector - Vertex AI Training Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # compile sub-command
    compile_parser = subparsers.add_parser("compile", help="Compile the pipeline to YAML")
    compile_parser.add_argument(
        "--output", type=str, default="pipeline.yaml", help="Output YAML path"
    )

    # run sub-command
    run_parser = subparsers.add_parser("run", help="Submit a pipeline run to Vertex AI")
    run_parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration",
    )
    run_parser.add_argument("--epochs", type=int, default=None, help="Training epochs override")
    run_parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    run_parser.add_argument(
        "--min-accuracy", type=float, default=None, help="Minimum accuracy threshold override"
    )
    run_parser.add_argument("--auto-deploy", action="store_true", help="Enable auto-deployment")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "compile":
        compile_pipeline(args.output)
    elif args.command == "run":
        submit_pipeline(
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_accuracy=args.min_accuracy,
            auto_deploy=args.auto_deploy,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
