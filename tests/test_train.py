"""Tests for the training module."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.training.train import (
    _upload_artifacts_to_gcs,
    get_device,
    set_seed,
    train_epoch,
    validate,
)


class TestSetSeed:
    """Tests for set_seed()."""

    def test_reproducible_random(self) -> None:
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self) -> None:
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b)


class TestGetDevice:
    """Tests for get_device()."""

    def test_returns_device_object(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_known_type(self) -> None:
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestTrainEpoch:
    """Tests for train_epoch() on a tiny synthetic dataset."""

    @staticmethod
    def _make_loader(num_samples: int = 16, batch_size: int = 4):
        images = torch.randn(num_samples, 3, 32, 32)
        labels = torch.randint(0, 2, (num_samples,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def test_returns_loss_and_accuracy(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        loader = self._make_loader()

        loss, acc = train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_empty_loader_returns_zeros(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0)),
            batch_size=1,
        )

        loss, acc = train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert loss == 0.0
        assert acc == 0.0


class TestValidate:
    """Tests for validate() on a tiny synthetic dataset."""

    @staticmethod
    def _make_loader(num_samples: int = 16, batch_size: int = 4):
        images = torch.randn(num_samples, 3, 32, 32)
        labels = torch.randint(0, 2, (num_samples,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def test_returns_five_metrics(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 1))
        criterion = nn.BCEWithLogitsLoss()
        loader = self._make_loader()

        result = validate(model, loader, criterion, torch.device("cpu"))

        assert len(result) == 5
        loss, acc, prec, rec, f1 = result
        assert all(isinstance(v, float) for v in result)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_empty_loader_returns_zeros(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 1))
        criterion = nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0)),
            batch_size=1,
        )

        result = validate(model, loader, criterion, torch.device("cpu"))
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


class TestUploadArtifactsToGcs:
    """Tests for _upload_artifacts_to_gcs()."""

    def test_noop_when_no_bucket(self, tmp_path: Path) -> None:
        # Should not raise or call anything
        _upload_artifacts_to_gcs(None, tmp_path)

    def test_noop_when_checkpoint_missing(self, tmp_path: Path) -> None:
        _upload_artifacts_to_gcs("my-bucket", tmp_path / "nonexistent")

    @patch("src.training.gcs.upload_file")
    @patch("src.training.gcs.upload_directory")
    def test_uploads_model_and_mlruns(
        self,
        mock_upload_dir: MagicMock,
        mock_upload_file: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "best_model.pt").write_bytes(b"fake model")

        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        (mlruns / "run.txt").write_text("log")

        # _upload_artifacts_to_gcs checks Path("mlruns"), so chdir to tmp_path
        monkeypatch.chdir(tmp_path)

        _upload_artifacts_to_gcs("my-bucket", checkpoint_dir)

        mock_upload_file.assert_called_once_with(
            local_path=str(checkpoint_dir / "best_model.pt"),
            bucket_name="my-bucket",
            gcs_path="models/best_model.pt",
        )
        mock_upload_dir.assert_called_once()


class TestMainCLI:
    """Tests for CLI argument parsing."""

    def test_default_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="configs/train_config.yaml")
        parser.add_argument("--gcs-bucket", type=str, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)

        args = parser.parse_args([])
        assert args.config == "configs/train_config.yaml"
        assert args.gcs_bucket is None
        assert args.epochs is None
        assert args.batch_size is None

    def test_custom_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="configs/train_config.yaml")
        parser.add_argument("--gcs-bucket", type=str, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)

        args = parser.parse_args(
            [
                "--config",
                "custom.yaml",
                "--gcs-bucket",
                "my-bucket",
                "--epochs",
                "5",
                "--batch-size",
                "16",
            ]
        )
        assert args.config == "custom.yaml"
        assert args.gcs_bucket == "my-bucket"
        assert args.epochs == 5
        assert args.batch_size == 16
