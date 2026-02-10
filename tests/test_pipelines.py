"""Tests for pipeline modules (import-level and pure utility functions)."""

import json
from pathlib import Path

import pytest


class TestEvaluateModule:
    """Tests for src.pipelines.evaluate."""

    def test_imports_without_error(self) -> None:
        import src.pipelines.evaluate  # noqa: F401

    def test_compute_metrics(self) -> None:
        import numpy as np

        from src.pipelines.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.3])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["total_samples"] == 5
        assert metrics["positive_samples"] == 3
        assert metrics["negative_samples"] == 2
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_compute_metrics_perfect_predictions(self) -> None:
        import numpy as np

        from src.pipelines.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["confusion_matrix"]["tp"] == 2
        assert metrics["confusion_matrix"]["tn"] == 2
        assert metrics["confusion_matrix"]["fp"] == 0
        assert metrics["confusion_matrix"]["fn"] == 0

    def test_compare_to_baseline_missing_file(self, tmp_path: Path) -> None:
        from src.pipelines.evaluate import compare_to_baseline

        result = compare_to_baseline(
            {"accuracy": 0.9, "f1": 0.85},
            tmp_path / "nonexistent.json",
        )
        assert "error" in result

    def test_compare_to_baseline_improved(self, tmp_path: Path) -> None:
        from src.pipelines.evaluate import compare_to_baseline

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(
            json.dumps(
                {
                    "accuracy": 0.80,
                    "precision": 0.75,
                    "recall": 0.70,
                    "f1": 0.72,
                    "auc_roc": 0.85,
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

        assert result["is_improved"] is True
        assert result["delta"]["accuracy"] > 0
        assert result["delta"]["f1"] > 0

    def test_compare_to_baseline_regression(self, tmp_path: Path) -> None:
        from src.pipelines.evaluate import compare_to_baseline

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(
            json.dumps(
                {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.90,
                    "f1": 0.91,
                    "auc_roc": 0.97,
                }
            )
        )

        current = {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "f1": 0.72,
            "auc_roc": 0.85,
        }

        result = compare_to_baseline(current, baseline_file)

        assert result["is_improved"] is False
        assert result["delta"]["accuracy"] < 0
        assert result["delta"]["f1"] < 0


class TestTrainingPipelineModule:
    """Tests for src.pipelines.training_pipeline.

    KFP component decorators may fail at import time when the installed
    kfp version is incompatible. We test what we can and skip gracefully.
    """

    def _try_import(self):
        try:
            import src.pipelines.training_pipeline  # noqa: F401

            return src.pipelines.training_pipeline
        except TypeError:
            pytest.skip("KFP version incompatibility prevents module import")

    def test_imports_or_skips_gracefully(self) -> None:
        self._try_import()

    def test_constants_defined(self) -> None:
        mod = self._try_import()
        assert mod.PROJECT_ID
        assert mod.REGION
        assert mod.GCS_BUCKET.startswith("gs://")
        assert mod.PIPELINE_ROOT.startswith(mod.GCS_BUCKET)
        assert mod.ARTIFACT_REGISTRY
        assert mod.TRAINING_IMAGE
        assert mod.SERVICE_NAME

    def test_compile_pipeline_function_exists(self) -> None:
        mod = self._try_import()
        assert callable(mod.compile_pipeline)

    def test_training_pipeline_function_exists(self) -> None:
        mod = self._try_import()
        assert callable(mod.training_pipeline)
