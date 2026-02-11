"""Extended tests for the evaluation pipeline (src/pipelines/evaluate.py)."""

import json
from pathlib import Path

import numpy as np
import torch

from src.pipelines.evaluate import (
    compare_to_baseline,
    compute_metrics,
    load_model,
    plot_confusion_matrix,
    run_inference,
)
from src.training.model import AIImageDetector


def _create_checkpoint(path: Path, epoch: int = 5, val_accuracy: float = 0.9) -> None:
    """Create a minimal checkpoint file."""
    model = AIImageDetector(
        model_name="efficientnet_b0",
        pretrained=False,
        dropout=0.3,
    )
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_accuracy": val_accuracy,
        "config": {
            "model": {
                "name": "efficientnet_b0",
                "dropout": 0.3,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


class TestLoadModel:
    """Tests for load_model()."""

    def test_loads_model_from_checkpoint(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "model.pt"
        _create_checkpoint(ckpt_path)

        model = load_model(ckpt_path, torch.device("cpu"))
        assert isinstance(model, AIImageDetector)
        assert not model.training  # Should be in eval mode

    def test_model_produces_output(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "model.pt"
        _create_checkpoint(ckpt_path)

        model = load_model(ckpt_path, torch.device("cpu"))
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 1)


class TestRunInference:
    """Tests for run_inference()."""

    def test_returns_correct_shapes(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "model.pt"
        _create_checkpoint(ckpt_path)
        model = load_model(ckpt_path, torch.device("cpu"))

        # Create synthetic dataset
        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 2, (8,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        y_true, y_pred, y_prob = run_inference(model, loader, torch.device("cpu"))

        assert len(y_true) == 8
        assert len(y_pred) == 8
        assert len(y_prob) == 8
        assert set(np.unique(y_pred)).issubset({0, 1})
        assert all(0.0 <= p <= 1.0 for p in y_prob)


class TestComputeMetricsExtended:
    """Extended tests for compute_metrics()."""

    def test_all_wrong_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 0.0
        assert metrics["confusion_matrix"]["tp"] == 0
        assert metrics["confusion_matrix"]["tn"] == 0

    def test_mostly_correct_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 < metrics["accuracy"] < 1.0
        assert metrics["positive_samples"] == 3
        assert metrics["negative_samples"] == 3

    def test_classification_report_included(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert "classification_report" in metrics
        assert "real" in metrics["classification_report"]
        assert "ai_generated" in metrics["classification_report"]


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix()."""

    def test_creates_png_file(self, tmp_path: Path) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        output_path = tmp_path / "cm.png"

        plot_confusion_matrix(y_true, y_pred, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_perfect_predictions(self, tmp_path: Path) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        output_path = tmp_path / "cm_perfect.png"

        plot_confusion_matrix(y_true, y_pred, output_path)
        assert output_path.exists()


class TestCompareToBaselineExtended:
    """Extended tests for compare_to_baseline()."""

    def test_equal_metrics(self, tmp_path: Path) -> None:
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(
            json.dumps(
                {
                    "accuracy": 0.90,
                    "precision": 0.88,
                    "recall": 0.85,
                    "f1": 0.86,
                    "auc_roc": 0.92,
                }
            )
        )

        current = {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.85,
            "f1": 0.86,
            "auc_roc": 0.92,
        }

        result = compare_to_baseline(current, baseline_file)
        assert result["is_improved"] is True  # Equal counts as improved
        assert result["delta"]["accuracy"] == 0.0
        assert result["delta"]["f1"] == 0.0

    def test_accuracy_improved_f1_regressed(self, tmp_path: Path) -> None:
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(
            json.dumps(
                {
                    "accuracy": 0.80,
                    "precision": 0.75,
                    "recall": 0.70,
                    "f1": 0.90,
                    "auc_roc": 0.85,
                }
            )
        )

        current = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.72,
            "f1": 0.75,
            "auc_roc": 0.87,
        }

        result = compare_to_baseline(current, baseline_file)
        assert result["is_improved"] is False
        assert result["delta"]["accuracy"] > 0
        assert result["delta"]["f1"] < 0
