"""Configuration management utilities."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data configuration."""

    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val"
    test_dir: str = "data/processed/test"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 1
    dropout: float = 0.3


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    early_stopping_patience: int = 5


class ThresholdsConfig(BaseModel):
    """Classification thresholds."""

    classification: float = 0.5
    high_confidence: float = 0.8
    low_confidence: float = 0.3


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_path: str = "models/checkpoints/best_model.pt"
    mlflow_tracking_uri: str = "http://localhost:5000"
    log_level: str = "INFO"

    model_config = ConfigDict(env_prefix="", case_sensitive=False)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary with configuration values.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings instance with environment variables.
    """
    return Settings()
