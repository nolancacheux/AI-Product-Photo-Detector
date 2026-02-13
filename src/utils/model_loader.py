"""Shared model loading utilities.

Centralizes checkpoint loading and classifier reconstruction logic used by
predictor, explainer, evaluate, and training_pipeline modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.training.model import AIImageDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _rebuild_classifier(
    state_dict: dict[str, Any],
    feature_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Rebuild classifier head to match checkpoint structure.

    Inspects state_dict keys to determine the original classifier layout,
    handling older checkpoints that lack BatchNorm1d.

    Args:
        state_dict: Model state dict from checkpoint.
        feature_dim: Number of input features from backbone.
        dropout: Dropout probability.

    Returns:
        Sequential classifier matching the checkpoint structure.
    """
    cls_keys = [k for k in state_dict if k.startswith("classifier.")]
    has_batchnorm = any(".running_mean" in k or ".running_var" in k for k in cls_keys)

    if has_batchnorm:
        return nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
        )
    return nn.Sequential(
        nn.Linear(feature_dim, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 1),
    )


def load_checkpoint(
    model_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a raw checkpoint from disk.

    Args:
        model_path: Path to the .pt checkpoint file.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    return torch.load(model_path, map_location=device, weights_only=False)


def load_model(
    model_path: str | Path,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> tuple[AIImageDetector, dict[str, Any]]:
    """Load a trained AIImageDetector from a checkpoint.

    Handles old checkpoints (without BatchNorm) by automatically
    rebuilding the classifier head to match the state_dict.

    Args:
        model_path: Path to the .pt checkpoint file.
        device: Target device (string or torch.device).
        eval_mode: Whether to set the model to eval mode.

    Returns:
        Tuple of (model, checkpoint_dict).

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if isinstance(device, str):
        device = torch.device(device)

    checkpoint = load_checkpoint(model_path, device)

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    model_name = model_config.get("name", "efficientnet_b0")
    dropout = model_config.get("dropout", 0.3)

    model = AIImageDetector(
        model_name=model_name,
        pretrained=False,
        dropout=dropout,
    )

    state_dict = checkpoint["model_state_dict"]

    # Try loading directly first; rebuild classifier on mismatch
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.classifier = _rebuild_classifier(state_dict, model.feature_dim, dropout)
        model.load_state_dict(state_dict)

    model = model.to(device)

    if eval_mode:
        model.eval()

    logger.info(
        "Model loaded",
        model_path=str(model_path),
        device=str(device),
        epoch=checkpoint.get("epoch", -1),
    )

    return model, checkpoint
