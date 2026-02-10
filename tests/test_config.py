"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.utils.config import (
    DataConfig,
    ModelConfig,
    Settings,
    ThresholdsConfig,
    TrainingConfig,
    get_settings,
    load_yaml_config,
)


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_load_valid_yaml(self, tmp_path) -> None:
        """Should load a valid YAML file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("key: value\nnested:\n  a: 1\n  b: 2\n")

        result = load_yaml_config(config_path)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1

    def test_load_nonexistent_raises(self, tmp_path) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml_config(tmp_path / "missing.yaml")

    def test_load_string_path(self, tmp_path) -> None:
        """Should accept string path."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("x: 42\n")

        result = load_yaml_config(str(config_path))
        assert result["x"] == 42


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self) -> None:
        """Settings should have sensible defaults."""
        settings = Settings()
        # MODEL_PATH is set in conftest.py, so just check it's a string
        assert isinstance(settings.model_path, str)
        assert settings.model_path.endswith(".pt")

    def test_settings_from_env(self) -> None:
        """Settings should read from environment variables."""
        with patch.dict(os.environ, {"AIDETECT_MODEL_PATH": "/custom/model.pt", "AIDETECT_LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.model_path == "/custom/model.pt"
            assert settings.log_level == "DEBUG"

    def test_get_settings(self) -> None:
        """get_settings should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)


class TestDataConfig:
    """Tests for DataConfig defaults."""

    def test_defaults(self) -> None:
        config = DataConfig()
        assert config.image_size == 224
        assert config.batch_size == 32
        assert config.num_workers == 4


class TestModelConfig:
    """Tests for ModelConfig defaults."""

    def test_defaults(self) -> None:
        config = ModelConfig()
        assert config.name == "efficientnet_b0"
        assert config.pretrained is True
        assert config.dropout == 0.3


class TestTrainingConfig:
    """Tests for TrainingConfig defaults."""

    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.epochs == 20
        assert config.learning_rate == 0.001


class TestThresholdsConfig:
    """Tests for ThresholdsConfig defaults."""

    def test_defaults(self) -> None:
        config = ThresholdsConfig()
        assert config.classification == 0.5
        assert config.high_confidence == 0.8
        assert config.low_confidence == 0.3
