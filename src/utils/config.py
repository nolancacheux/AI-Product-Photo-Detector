"""Configuration management utilities."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


class LogLevel(StrEnum):
    """Valid log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataConfig(BaseModel):
    """Data configuration."""

    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val"
    test_dir: str = "data/processed/test"
    image_size: int = Field(default=224, gt=0)
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.3, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    scheduler: str = "cosine"
    warmup_epochs: int = Field(default=2, ge=0)
    early_stopping_patience: int = Field(default=5, gt=0)


class ThresholdsConfig(BaseModel):
    """Classification thresholds."""

    classification: float = Field(default=0.5, ge=0.0, le=1.0)
    high_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    low_confidence: float = Field(default=0.3, ge=0.0, le=1.0)


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_path: str = "models/checkpoints/best_model.pt"
    mlflow_tracking_uri: str = "http://localhost:5000"
    log_level: LogLevel = LogLevel.INFO

    model_config = ConfigDict(env_prefix="AIDETECT_", case_sensitive=False)  # type: ignore[typeddict-unknown-key,assignment]


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
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings instance with environment variables.
    """
    return Settings()
