"""Training pipeline for AI Product Photo Detector."""

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
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            }
        )

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Validate the model.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (loss, accuracy, precision, recall).
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
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # For precision/recall
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    return avg_loss, accuracy, precision, recall


def train(config_path: str) -> None:
    """Main training function.

    Args:
        config_path: Path to training configuration file.
    """
    # Load configuration
    config = load_yaml_config(config_path)

    # Setup
    setup_logging(level="INFO", json_format=False)
    set_seed(config.get("seed", 42))
    device = get_device()

    logger.info("Starting training", device=str(device))

    # Data
    data_config = config.get("data", {})
    train_loader, val_loader = create_dataloaders(
        train_dir=data_config.get("train_dir", "data/processed/train"),
        val_dir=data_config.get("val_dir", "data/processed/val"),
        batch_size=data_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        image_size=data_config.get("image_size", 224),
    )

    logger.info(
        "Data loaded",
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
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
    criterion = nn.BCELoss()
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

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_config.get("name"),
                "learning_rate": training_config.get("learning_rate"),
                "batch_size": data_config.get("batch_size"),
                "epochs": epochs,
                "seed": config.get("seed"),
            }
        )

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss, val_acc, val_precision, val_recall = validate(
                model, val_loader, criterion, device
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
                    "learning_rate": scheduler.get_last_lr()[0],
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

                # Save checkpoint
                checkpoint_dir = Path(
                    config.get("checkpoint", {}).get("save_dir", "models/checkpoints")
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "config": config,
                    },
                    checkpoint_dir / "best_model.pt",
                )

                logger.info("Saved best model", val_accuracy=f"{val_acc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Log final model
        if mlflow_config.get("log_models", True):
            mlflow.pytorch.log_model(model, "model")

        logger.info("Training complete", best_val_accuracy=f"{best_val_accuracy:.4f}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train AI Product Photo Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file",
    )
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
