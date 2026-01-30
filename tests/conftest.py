"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Set environment variables for testing
os.environ["MODEL_PATH"] = "models/test_model.pt"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """Get path to models directory."""
    return Path(__file__).parent.parent / "models"
