"""Extended tests for the training pipeline module (src/pipelines/training_pipeline.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _try_import_module():
    """Import the training_pipeline module, skip if KFP is incompatible."""
    try:
        import src.pipelines.training_pipeline as mod

        return mod
    except TypeError:
        pytest.skip("KFP version incompatibility prevents module import")


class TestCompilePipeline:
    """Tests for compile_pipeline()."""

    def test_compile_creates_yaml_file(self, tmp_path: Path) -> None:
        mod = _try_import_module()
        output_path = str(tmp_path / "pipeline.yaml")
        mod.compile_pipeline(output_path)
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert len(content) > 0

    def test_compile_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _try_import_module()
        monkeypatch.chdir(tmp_path)
        mod.compile_pipeline()
        assert (tmp_path / "pipeline.yaml").exists()


class TestSubmitPipeline:
    """Tests for submit_pipeline()."""

    def test_submit_calls_aiplatform(self) -> None:
        mod = _try_import_module()

        with (
            patch.object(mod, "load_yaml_config") as mock_config,
            patch.object(mod, "compile_pipeline"),
            patch("google.cloud.aiplatform.init"),
            patch("google.cloud.aiplatform.PipelineJob") as mock_job_cls,
        ):
            mock_config.return_value = {
                "pipeline": {},
                "training": {},
                "evaluation": {},
                "deployment": {},
            }
            mock_job = MagicMock()
            mock_job.resource_name = "test-resource"
            mock_job_cls.return_value = mock_job

            mod.submit_pipeline(config_path="test.yaml", epochs=5, batch_size=32)

            mock_job.submit.assert_called_once()


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    def test_compile_command(self) -> None:
        mod = _try_import_module()

        with (
            patch.object(mod, "compile_pipeline") as mock_compile,
            patch("sys.argv", ["training_pipeline", "compile", "--output", "out.yaml"]),
        ):
            mod.main()
            mock_compile.assert_called_once_with("out.yaml")

    def test_run_command(self) -> None:
        mod = _try_import_module()

        with (
            patch.object(mod, "submit_pipeline") as mock_submit,
            patch(
                "sys.argv",
                ["training_pipeline", "run", "--config", "test.yaml", "--epochs", "10"],
            ),
        ):
            mod.main()
            mock_submit.assert_called_once()
            call_kwargs = mock_submit.call_args[1]
            assert call_kwargs["config_path"] == "test.yaml"
            assert call_kwargs["epochs"] == 10

    def test_no_command_prints_help(self) -> None:
        mod = _try_import_module()

        with patch("sys.argv", ["training_pipeline"]):
            mod.main()  # Should not raise
