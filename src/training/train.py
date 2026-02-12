"""Training pipeline for AI Product Photo Detector.

Supports both local and Vertex AI (GCS-backed) training. When a --gcs-bucket
is provided, data is pulled from GCS before training and the best model +
MLflow artifacts are uploaded to GCS after training.
"""

import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.dataset import create_dataloaders
from src.training.model import create_model
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device (CUDA, MPS, or CPU).

    Returns:
        torch.device for training.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics (threshold at 0.0 for logits, equivalent to 0.5 after sigmoid)
        total_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.0).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            }
        )

    if total == 0:
        logger.warning("Training loader was empty -- no samples processed")
        return 0.0, 0.0

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Validate the model.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (loss, accuracy, precision, recall, f1).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # For precision/recall
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    if total == 0:
        logger.warning("Validation loader was empty -- no samples processed")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, accuracy, precision, recall, f1


def train(
    config_path: str,
    gcs_bucket: str | None = None,
    epoch_override: int | None = None,
    batch_size_override: int | None = None,
) -> None:
    """Main training function.

    Args:
        config_path: Path to training configuration file.
        gcs_bucket: GCS bucket name for remote data/model storage.
            When set, data is downloaded from GCS if missing locally,
            and the best model is uploaded to GCS after training.
        epoch_override: Override the number of epochs from config.
        batch_size_override: Override the batch size from config.
    """
    # Load configuration
    config = load_yaml_config(config_path)

    # Apply CLI overrides
    if epoch_override is not None:
        config.setdefault("training", {})["epochs"] = epoch_override
    if batch_size_override is not None:
        config.setdefault("data", {})["batch_size"] = batch_size_override

    # Setup
    setup_logging(level="INFO", json_format=False)
    set_seed(config.get("seed", 42))
    device = get_device()

    logger.info("Starting training", device=str(device), gcs_bucket=gcs_bucket)

    # GCS data download: pull training data if local dirs are missing
    data_config = config.get("data", {})
    train_dir = data_config.get("train_dir", "data/processed/train")
    val_dir = data_config.get("val_dir", "data/processed/val")

    if gcs_bucket and (not Path(train_dir).exists() or not Path(val_dir).exists()):
        from src.training.gcs import download_directory

        logger.info("Local data not found, downloading from GCS", bucket=gcs_bucket)
        download_directory(
            bucket_name=gcs_bucket,
            gcs_prefix="data/processed",
            local_dir="data/processed",
        )

    # Data
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=data_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        image_size=data_config.get("image_size", 224),
    )

    logger.info(
        "Data loaded",
        train_samples=len(train_loader.dataset),  # type: ignore[arg-type]
        val_samples=len(val_loader.dataset),  # type: ignore[arg-type]
    )

    # Model
    model_config = config.get("model", {})
    model = create_model(
        model_name=model_config.get("name", "efficientnet_b0"),
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", 0.3),
    )
    model = model.to(device)

    logger.info(
        "Model created",
        model_name=model_config.get("name"),
        trainable_params=model.get_num_trainable_params(),
    )

    # Training setup
    training_config = config.get("training", {})
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0001),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config.get("epochs", 20),
    )

    # MLflow tracking
    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_config.get("experiment_name", "ai-product-detector"))

    # Training loop
    best_val_accuracy = 0.0
    epochs = training_config.get("epochs", 20)
    patience = training_config.get("early_stopping_patience", 5)
    patience_counter = 0
    checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "models/checkpoints"))

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_config.get("name"),
                "learning_rate": training_config.get("learning_rate"),
                "weight_decay": training_config.get("weight_decay"),
                "batch_size": data_config.get("batch_size"),
                "image_size": data_config.get("image_size"),
                "epochs": epochs,
                "seed": config.get("seed"),
                "dropout": model_config.get("dropout"),
                "pretrained": model_config.get("pretrained"),
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "early_stopping_patience": patience,
                "num_workers": data_config.get("num_workers"),
                "trainable_params": model.get_num_trainable_params(),
                "total_params": model.get_num_total_params(),
                "device": str(device),
                "gcs_bucket": gcs_bucket or "none",
            }
        )

        for epoch in range(epochs):
            logger.info("Epoch started", epoch=epoch + 1, total_epochs=epochs)

            # Train
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
                model,
                val_loader,
                criterion,
                device,
            )

            # Step scheduler
            scheduler.step()

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                },
                step=epoch,
            )

            logger.info(
                "Epoch complete",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
            )

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0

                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_accuracy": val_acc,
                        "best_val_accuracy": best_val_accuracy,
                        "config": config,
                    },
                    checkpoint_dir / "best_model.pt",
                )

                mlflow.log_artifact(str(checkpoint_dir / "best_model.pt"))
                logger.info("Saved best model", val_accuracy=f"{val_acc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping triggered", epoch=epoch + 1)
                break

        # Log best metrics
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)

        # Log final model
        if mlflow_config.get("log_models", True):
            mlflow.pytorch.log_model(model, "model")

        # Log config as artifact
        mlflow.log_artifact(config_path)

        logger.info("Training complete", best_val_accuracy=f"{best_val_accuracy:.4f}")

    # Upload artifacts to GCS after training
    _upload_artifacts_to_gcs(gcs_bucket, checkpoint_dir)


def _upload_artifacts_to_gcs(
    gcs_bucket: str | None,
    checkpoint_dir: Path,
) -> None:
    """Upload model checkpoint and MLflow artifacts to GCS.

    No-op when gcs_bucket is None (local-only training).

    Args:
        gcs_bucket: GCS bucket name, or None for local training.
        checkpoint_dir: Local directory containing the best model checkpoint.
    """
    best_model_path = checkpoint_dir / "best_model.pt"
    if not gcs_bucket or not best_model_path.exists():
        return

    from src.training.gcs import upload_directory, upload_file

    logger.info("Uploading best model to GCS")
    upload_file(
        local_path=str(best_model_path),
        bucket_name=gcs_bucket,
        gcs_path="models/best_model.pt",
    )

    # Upload MLflow run artifacts
    mlruns_dir = Path("mlruns")
    if mlruns_dir.exists():
        logger.info("Uploading MLflow artifacts to GCS")
        upload_directory(
            local_dir=str(mlruns_dir),
            bucket_name=gcs_bucket,
            gcs_prefix="mlruns",
        )

    logger.info("GCS upload complete")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train AI Product Photo Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket for remote data/model storage (e.g. ai-product-detector-487013)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        gcs_bucket=args.gcs_bucket,
        epoch_override=args.epochs,
        batch_size_override=args.batch_size,
    )


if __name__ == "__main__":
    main()
