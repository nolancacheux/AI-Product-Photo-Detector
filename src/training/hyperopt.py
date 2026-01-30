"""Hyperparameter optimization with Optuna."""

import argparse
from pathlib import Path
from typing import Any

import mlflow
import optuna
import torch
import torch.nn as nn
from optuna.integration.mlflow import MLflowCallback
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.augmentation import MixUpCutMixCollator, get_advanced_train_transforms
from src.training.dataset import AIProductDataset, get_val_transforms
from src.training.model import create_model
from src.training.train import get_device, set_seed, train_epoch, validate
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def create_objective(
    train_dir: str,
    val_dir: str,
    n_epochs: int = 10,
    seed: int = 42,
) -> callable:
    """Create Optuna objective function.

    Args:
        train_dir: Training data directory.
        val_dir: Validation data directory.
        n_epochs: Number of epochs per trial.
        seed: Random seed.

    Returns:
        Objective function for Optuna.
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial.

        Returns:
            Validation accuracy (to maximize).
        """
        set_seed(seed)
        device = get_device()

        # Hyperparameters to tune
        model_name = trial.suggest_categorical(
            "model_name",
            ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        # Augmentation parameters
        use_randaugment = trial.suggest_categorical("use_randaugment", [True, False])
        randaugment_n = trial.suggest_int("randaugment_n", 1, 3) if use_randaugment else 2
        randaugment_m = trial.suggest_int("randaugment_m", 5, 15) if use_randaugment else 9

        use_mixup_cutmix = trial.suggest_categorical("use_mixup_cutmix", [True, False])
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.4) if use_mixup_cutmix else 0.2
        cutmix_alpha = trial.suggest_float("cutmix_alpha", 0.5, 1.5) if use_mixup_cutmix else 1.0

        # Create datasets
        train_transform = get_advanced_train_transforms(
            image_size=224,
            use_randaugment=use_randaugment,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
        )

        train_dataset = AIProductDataset(train_dir, transform=train_transform)
        val_dataset = AIProductDataset(val_dir, transform=get_val_transforms(224))

        # Create collator for MixUp/CutMix
        collate_fn = None
        if use_mixup_cutmix:
            collate_fn = MixUpCutMixCollator(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Create model
        model = create_model(
            model_name=model_name,
            pretrained=True,
            dropout=dropout,
        )
        model = model.to(device)

        # Training setup
        criterion = nn.BCELoss()
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Training loop
        best_val_accuracy = 0.0

        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_precision, val_recall = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            # Report intermediate value
            trial.report(val_acc, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc

            logger.info(
                f"Trial {trial.number} - Epoch {epoch + 1}/{n_epochs}",
                val_accuracy=f"{val_acc:.4f}",
                best=f"{best_val_accuracy:.4f}",
            )

        return best_val_accuracy

    return objective


def run_hyperopt(
    config_path: str,
    n_trials: int = 50,
    timeout: int | None = None,
    study_name: str = "ai-detector-hyperopt",
    storage: str | None = None,
) -> optuna.Study:
    """Run hyperparameter optimization.

    Args:
        config_path: Path to base config file.
        n_trials: Number of trials to run.
        timeout: Timeout in seconds.
        study_name: Name of the Optuna study.
        storage: Optuna storage URL (e.g., sqlite:///optuna.db).

    Returns:
        Completed Optuna study.
    """
    setup_logging(level="INFO", json_format=False)

    # Load config
    config = load_yaml_config(config_path)
    data_config = config.get("data", {})

    train_dir = data_config.get("train_dir", "data/processed/train")
    val_dir = data_config.get("val_dir", "data/processed/val")
    n_epochs = config.get("hyperopt", {}).get("epochs_per_trial", 10)
    seed = config.get("seed", 42)

    logger.info(
        "Starting hyperparameter optimization",
        n_trials=n_trials,
        epochs_per_trial=n_epochs,
    )

    # Create study
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # MLflow callback
    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "http://localhost:5000"))

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow_config.get("tracking_uri", "http://localhost:5000"),
        metric_name="val_accuracy",
        mlflow_kwargs={"nested": True},
    )

    # Run optimization
    objective = create_objective(train_dir, val_dir, n_epochs, seed)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )

    # Log results
    logger.info(
        "Hyperparameter optimization complete",
        best_trial=study.best_trial.number,
        best_value=f"{study.best_value:.4f}",
        best_params=study.best_params,
    )

    return study


def get_best_config(study: optuna.Study) -> dict[str, Any]:
    """Convert best trial parameters to training config.

    Args:
        study: Completed Optuna study.

    Returns:
        Training configuration dictionary.
    """
    params = study.best_params

    return {
        "model": {
            "name": params["model_name"],
            "pretrained": True,
            "dropout": params["dropout"],
        },
        "training": {
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "epochs": 50,  # Full training
        },
        "data": {
            "batch_size": params["batch_size"],
        },
        "augmentation": {
            "use_randaugment": params["use_randaugment"],
            "randaugment_n": params.get("randaugment_n", 2),
            "randaugment_m": params.get("randaugment_m", 9),
            "use_mixup_cutmix": params["use_mixup_cutmix"],
            "mixup_alpha": params.get("mixup_alpha", 0.2),
            "cutmix_alpha": params.get("cutmix_alpha", 1.0),
        },
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_value,
        },
    }


def main() -> None:
    """CLI entry point for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to base training configuration",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ai-detector-hyperopt",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/best_config.yaml",
        help="Output path for best config",
    )

    args = parser.parse_args()

    # Run optimization
    study = run_hyperopt(
        config_path=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
    )

    # Save best config
    best_config = get_best_config(study)

    import yaml

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    logger.info(f"Best config saved to {output_path}")


if __name__ == "__main__":
    main()
