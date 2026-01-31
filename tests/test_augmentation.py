"""Tests for data augmentation module."""

import torch
from PIL import Image

from src.training.augmentation import get_train_transforms, get_val_transforms


class TestTrainTransforms:
    """Tests for training transforms."""

    def test_output_shape(self) -> None:
        """Test output shape is correct."""
        transforms = get_train_transforms(image_size=224)
        img = Image.new("RGB", (256, 256), color="red")
        result = transforms(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_different_sizes(self) -> None:
        """Test with different image sizes."""
        for size in [128, 224, 384]:
            transforms = get_train_transforms(image_size=size)
            img = Image.new("RGB", (512, 512), color="blue")
            result = transforms(img)
            assert result.shape == (3, size, size)


class TestValTransforms:
    """Tests for validation transforms."""

    def test_output_shape(self) -> None:
        """Test output shape is correct."""
        transforms = get_val_transforms(image_size=224)
        img = Image.new("RGB", (256, 256), color="green")
        result = transforms(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_deterministic(self) -> None:
        """Test validation transforms are deterministic."""
        transforms = get_val_transforms(image_size=224)
        img = Image.new("RGB", (256, 256), color="red")

        result1 = transforms(img)
        result2 = transforms(img)

        assert torch.allclose(result1, result2)
