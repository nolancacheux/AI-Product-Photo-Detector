"""Model evaluation for AI Product Photo Detector.

Loads a trained checkpoint, runs inference on the test set, and computes
comprehensive classification metrics. Results are persisted as JSON and
optionally compared against a baseline metrics file.

Usage:
    python -m src.pipelines.evaluate \
        --model-path models/checkpoints/best_model.pt \
        --test-dir data/processed/test \
        --output-dir reports \
        --baseline-metrics reports/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.training.dataset import AIProductDataset
from src.training.model import AIImageDetector
from src.utils.model_loader import load_model as _load_model_from_checkpoint

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["real", "ai_generated"]


def load_model(model_path: str | Path, device: torch.device) -> AIImageDetector:
    """Load a trained model from a checkpoint file.

    Args:
        model_path: Path to the .pt checkpoint.
        device: Target device.

    Returns:
        Model loaded with trained weights, in eval mode.
    """
    model, checkpoint = _load_model_from_checkpoint(
        model_path, device=device, eval_mode=True
    )

    logger.info(
        "Loaded model from %s (epoch %d, val_accuracy=%.4f)",
        model_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_accuracy", 0.0),
    )
    return model


def run_inference(
    model: AIImageDetector,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a dataset and collect predictions.

    Args:
        model: Trained model in eval mode.
        data_loader: DataLoader for the evaluation set.
        device: Target device.

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities).
    """
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric names to values.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": roc_auc,
        "total_samples": int(len(y_true)),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int((y_true == 0).sum()),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        output_path: Destination file path for the PNG.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts and percentages
    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def compare_to_baseline(
    current: dict[str, Any],
    baseline_path: str | Path,
) -> dict[str, Any]:
    """Compare current metrics against a baseline file.

    Args:
        current: Current evaluation metrics dictionary.
        baseline_path: Path to the baseline metrics JSON.

    Returns:
        Comparison dictionary with deltas and an ``is_improved`` flag.
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        logger.warning("Baseline file not found: %s", baseline_path)
        return {"error": f"Baseline file not found: {baseline_path}"}

    with open(baseline_path) as f:
        baseline = json.load(f)

    compare_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    comparison: dict[str, Any] = {"baseline": {}, "current": {}, "delta": {}}

    for key in compare_keys:
        b_val = baseline.get(key, 0.0)
        c_val = current.get(key, 0.0)
        comparison["baseline"][key] = b_val
        comparison["current"][key] = c_val
        comparison["delta"][key] = round(c_val - b_val, 6)

    # Model is considered improved if both accuracy and F1 are at least as good
    comparison["is_improved"] = (
        comparison["delta"]["accuracy"] >= 0 and comparison["delta"]["f1"] >= 0
    )

    return comparison


def evaluate(
    model_path: str,
    test_dir: str,
    output_dir: str,
    baseline_metrics: str | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
) -> dict[str, Any]:
    """Full evaluation pipeline.

    Args:
        model_path: Path to model checkpoint.
        test_dir: Path to test data directory.
        output_dir: Directory for output artifacts.
        baseline_metrics: Optional path to baseline metrics JSON.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        image_size: Input image resolution.

    Returns:
        Complete metrics dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    model = load_model(model_path, device)

    # Build test loader
    test_dataset = AIProductDataset(
        data_dir=test_dir,
        image_size=image_size,
    )

    if len(test_dataset) == 0:
        logger.error("No test samples found in %s", test_dir)
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("Test dataset: %d samples", len(test_dataset))

    # Run inference
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info(
        "Results: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, auc_roc=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc_roc"],
    )

    # Save confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, output_path / "confusion_matrix.png")

    # Compare to baseline if provided
    if baseline_metrics:
        comparison = compare_to_baseline(metrics, baseline_metrics)
        metrics["baseline_comparison"] = comparison
        logger.info("Baseline comparison: %s", json.dumps(comparison, indent=2))

    # Save metrics JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_file)

    return metrics


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate AI Product Photo Detector model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/processed/test",
        help="Path to test dataset directory (default: data/processed/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output artifacts (default: reports)",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=None,
        help="Path to baseline metrics JSON for comparison",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image resolution (default: 224)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluate(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        baseline_metrics=args.baseline_metrics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
