"""Tests for Vertex AI training submission (src/training/vertex_submit.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_constants_defined(self) -> None:
        from src.training.vertex_submit import (
            ARTIFACT_REGISTRY,
            GCS_BUCKET,
            IMAGE_NAME,
            IMAGE_TAG,
            IMAGE_URI,
            PROJECT_ID,
            REGION,
        )

        assert PROJECT_ID == "ai-product-detector-487013"
        assert REGION == "europe-west1"
        assert GCS_BUCKET == "ai-product-detector-487013"
        assert IMAGE_NAME == "ai-product-detector-train"
        assert IMAGE_TAG == "latest"
        assert ARTIFACT_REGISTRY in IMAGE_URI
        assert IMAGE_NAME in IMAGE_URI


class TestBuildAndPushImage:
    """Tests for build_and_push_image()."""

    @patch("src.training.vertex_submit.subprocess.run")
    def test_runs_three_commands(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from src.training.vertex_submit import build_and_push_image

        mock_run.return_value = MagicMock(returncode=0)

        result = build_and_push_image(tmp_path)

        assert mock_run.call_count == 3
        assert isinstance(result, str)
        assert "ai-product-detector-train" in result

    @patch("src.training.vertex_submit.subprocess.run")
    def test_docker_auth_called_first(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from src.training.vertex_submit import build_and_push_image

        mock_run.return_value = MagicMock(returncode=0)
        build_and_push_image(tmp_path)

        first_call_args = mock_run.call_args_list[0]
        assert "gcloud" in first_call_args[0][0]
        assert "configure-docker" in first_call_args[0][0]

    @patch("src.training.vertex_submit.subprocess.run")
    def test_docker_build_called(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from src.training.vertex_submit import build_and_push_image

        mock_run.return_value = MagicMock(returncode=0)
        build_and_push_image(tmp_path)

        second_call_args = mock_run.call_args_list[1]
        assert "docker" in second_call_args[0][0]
        assert "build" in second_call_args[0][0]

    @patch("src.training.vertex_submit.subprocess.run")
    def test_docker_push_called(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from src.training.vertex_submit import build_and_push_image

        mock_run.return_value = MagicMock(returncode=0)
        build_and_push_image(tmp_path)

        third_call_args = mock_run.call_args_list[2]
        assert "docker" in third_call_args[0][0]
        assert "push" in third_call_args[0][0]

    @patch("src.training.vertex_submit.subprocess.run", side_effect=Exception("docker fail"))
    def test_raises_on_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        from src.training.vertex_submit import build_and_push_image

        with pytest.raises(Exception, match="docker fail"):
            build_and_push_image(tmp_path)


class TestSubmitTrainingJob:
    """Tests for submit_training_job()."""

    @patch("src.training.vertex_submit.aiplatform")
    def test_submits_job(self, mock_aiplatform: MagicMock) -> None:
        from src.training.vertex_submit import submit_training_job

        mock_job = MagicMock()
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        result = submit_training_job(
            epochs=10,
            batch_size=32,
            config_path="configs/train_config.yaml",
            sync=False,
        )

        mock_aiplatform.init.assert_called_once()
        mock_aiplatform.CustomContainerTrainingJob.assert_called_once()
        mock_job.run.assert_called_once()
        assert result is mock_job

    @patch("src.training.vertex_submit.aiplatform")
    def test_passes_correct_args(self, mock_aiplatform: MagicMock) -> None:
        from src.training.vertex_submit import submit_training_job

        mock_job = MagicMock()
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        submit_training_job(epochs=5, batch_size=16, config_path="my.yaml", sync=True)

        run_kwargs = mock_job.run.call_args
        args_list = run_kwargs[1]["args"]
        assert "--epochs" in args_list
        assert "5" in args_list
        assert "--batch-size" in args_list
        assert "16" in args_list
        assert "--config" in args_list
        assert "my.yaml" in args_list
        assert run_kwargs[1]["sync"] is True


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    @patch("src.training.vertex_submit.submit_training_job")
    @patch("src.training.vertex_submit.build_and_push_image")
    @patch("src.training.vertex_submit.upload_directory", return_value=5)
    @patch("src.training.vertex_submit.setup_logging")
    def test_main_skip_upload_skip_build(
        self,
        mock_logging: MagicMock,
        mock_upload: MagicMock,
        mock_build: MagicMock,
        mock_submit: MagicMock,
    ) -> None:
        from src.training.vertex_submit import main

        mock_job = MagicMock()
        mock_job.display_name = "test-job"
        mock_submit.return_value = mock_job

        with patch(
            "sys.argv",
            ["vertex_submit", "--skip-upload", "--skip-build", "--epochs", "3"],
        ):
            main()

        mock_upload.assert_not_called()
        mock_build.assert_not_called()
        mock_submit.assert_called_once()

    @patch("src.training.vertex_submit.submit_training_job")
    @patch("src.training.vertex_submit.build_and_push_image")
    @patch("src.training.vertex_submit.upload_directory", return_value=10)
    @patch("src.training.vertex_submit.setup_logging")
    def test_main_with_upload_and_build(
        self,
        mock_logging: MagicMock,
        mock_upload: MagicMock,
        mock_build: MagicMock,
        mock_submit: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.vertex_submit import main

        # Create the expected data directory
        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)

        mock_job = MagicMock()
        mock_job.display_name = "test-job"
        mock_submit.return_value = mock_job

        with patch("src.training.vertex_submit.Path") as mock_path_cls:
            # Make Path(__file__).resolve().parents[2] return tmp_path
            mock_file_path = MagicMock()
            mock_file_path.resolve.return_value.parents.__getitem__ = lambda self, i: tmp_path
            mock_path_cls.return_value = mock_file_path
            mock_path_cls.__class__ = type(Path())

            with patch("sys.argv", ["vertex_submit", "--skip-upload", "--skip-build"]):
                main()

    @patch("src.training.vertex_submit.download_file")
    @patch("src.training.vertex_submit.submit_training_job")
    @patch("src.training.vertex_submit.build_and_push_image")
    @patch("src.training.vertex_submit.upload_directory", return_value=5)
    @patch("src.training.vertex_submit.setup_logging")
    def test_main_sync_downloads_model(
        self,
        mock_logging: MagicMock,
        mock_upload: MagicMock,
        mock_build: MagicMock,
        mock_submit: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        from src.training.vertex_submit import main

        mock_job = MagicMock()
        mock_submit.return_value = mock_job

        with patch(
            "sys.argv",
            ["vertex_submit", "--skip-upload", "--skip-build", "--sync"],
        ):
            main()

        mock_download.assert_called_once()

    @patch("src.training.vertex_submit.download_file", side_effect=Exception("GCS error"))
    @patch("src.training.vertex_submit.submit_training_job")
    @patch("src.training.vertex_submit.build_and_push_image")
    @patch("src.training.vertex_submit.upload_directory", return_value=5)
    @patch("src.training.vertex_submit.setup_logging")
    def test_main_sync_download_failure_exits(
        self,
        mock_logging: MagicMock,
        mock_upload: MagicMock,
        mock_build: MagicMock,
        mock_submit: MagicMock,
        mock_download: MagicMock,
    ) -> None:
        from src.training.vertex_submit import main

        mock_job = MagicMock()
        mock_submit.return_value = mock_job

        with patch(
            "sys.argv",
            ["vertex_submit", "--skip-upload", "--skip-build", "--sync"],
        ):
            with pytest.raises(SystemExit):
                main()
