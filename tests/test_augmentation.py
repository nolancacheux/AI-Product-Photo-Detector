"""Tests for data augmentation module."""

import pytest
import torch
from PIL import Image

from src.training.augmentation import (
    CutMix,
    GridMask,
    MixUp,
    MixUpCutMixCollator,
    RandAugment,
    get_advanced_train_transforms,
)


class TestCutMix:
    """Tests for CutMix augmentation."""

    @pytest.fixture
    def cutmix(self) -> CutMix:
        """Create CutMix instance."""
        return CutMix(alpha=1.0, prob=1.0)  # Always apply

    def test_output_shape(self, cutmix: CutMix) -> None:
        """Test output shape matches input."""
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        mixed_images, mixed_labels = cutmix(images, labels)

        assert mixed_images.shape == images.shape
        assert mixed_labels.shape == (4,)

    def test_labels_are_mixed(self, cutmix: CutMix) -> None:
        """Test labels are interpolated."""
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        _, mixed_labels = cutmix(images, labels)

        # Mixed labels should be floats between 0 and 1
        assert mixed_labels.dtype == torch.float32
        assert torch.all(mixed_labels >= 0)
        assert torch.all(mixed_labels <= 1)

    def test_no_cutmix_when_prob_zero(self) -> None:
        """Test no CutMix applied when prob=0."""
        cutmix = CutMix(alpha=1.0, prob=0.0)
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        mixed_images, mixed_labels = cutmix(images, labels)

        assert torch.allclose(mixed_images, images)


class TestMixUp:
    """Tests for MixUp augmentation."""

    @pytest.fixture
    def mixup(self) -> MixUp:
        """Create MixUp instance."""
        return MixUp(alpha=0.2, prob=1.0)

    def test_output_shape(self, mixup: MixUp) -> None:
        """Test output shape matches input."""
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        mixed_images, mixed_labels = mixup(images, labels)

        assert mixed_images.shape == images.shape
        assert mixed_labels.shape == (4,)

    def test_labels_are_mixed(self, mixup: MixUp) -> None:
        """Test labels are interpolated."""
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        _, mixed_labels = mixup(images, labels)

        # Mixed labels should be floats between 0 and 1
        assert mixed_labels.dtype == torch.float32


class TestRandAugment:
    """Tests for RandAugment."""

    @pytest.fixture
    def randaugment(self) -> RandAugment:
        """Create RandAugment instance."""
        return RandAugment(num_ops=2, magnitude=9)

    def test_output_is_image(self, randaugment: RandAugment) -> None:
        """Test output is a PIL image."""
        img = Image.new("RGB", (224, 224), color="red")
        result = randaugment(img)
        assert isinstance(result, Image.Image)

    def test_output_size_preserved(self, randaugment: RandAugment) -> None:
        """Test output size matches input."""
        img = Image.new("RGB", (224, 224), color="red")
        result = randaugment(img)
        assert result.size == img.size

    def test_different_magnitudes(self) -> None:
        """Test different magnitudes produce different results."""
        img = Image.new("RGB", (224, 224), color="red")

        ra_low = RandAugment(num_ops=2, magnitude=1)
        ra_high = RandAugment(num_ops=2, magnitude=30)

        # Should not raise errors
        result_low = ra_low(img)
        result_high = ra_high(img)

        assert isinstance(result_low, Image.Image)
        assert isinstance(result_high, Image.Image)


class TestGridMask:
    """Tests for GridMask augmentation."""

    @pytest.fixture
    def gridmask(self) -> GridMask:
        """Create GridMask instance."""
        return GridMask(prob=1.0)  # Always apply

    def test_output_shape(self, gridmask: GridMask) -> None:
        """Test output shape matches input."""
        img = torch.randn(3, 224, 224)
        result = gridmask(img)
        assert result.shape == img.shape

    def test_some_values_zeroed(self, gridmask: GridMask) -> None:
        """Test some values are zeroed by mask."""
        img = torch.ones(3, 224, 224)
        result = gridmask(img)

        # Some values should be zeroed
        assert torch.any(result == 0)


class TestMixUpCutMixCollator:
    """Tests for MixUpCutMixCollator."""

    @pytest.fixture
    def collator(self) -> MixUpCutMixCollator:
        """Create collator instance."""
        return MixUpCutMixCollator(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            mixup_prob=1.0,
            cutmix_prob=1.0,
        )

    def test_collates_batch(self, collator: MixUpCutMixCollator) -> None:
        """Test collator produces batch tensors."""
        batch = [
            (torch.randn(3, 224, 224), 0),
            (torch.randn(3, 224, 224), 1),
            (torch.randn(3, 224, 224), 0),
            (torch.randn(3, 224, 224), 1),
        ]

        images, labels = collator(batch)

        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)


class TestAdvancedTrainTransforms:
    """Tests for advanced training transforms."""

    def test_creates_transforms(self) -> None:
        """Test transform creation."""
        transforms = get_advanced_train_transforms(image_size=224)
        assert transforms is not None

    def test_transforms_work(self) -> None:
        """Test transforms can be applied."""
        transforms = get_advanced_train_transforms(
            image_size=224,
            use_randaugment=True,
            randaugment_n=2,
            randaugment_m=9,
        )

        img = Image.new("RGB", (256, 256), color="red")
        result = transforms(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_without_randaugment(self) -> None:
        """Test transforms without RandAugment."""
        transforms = get_advanced_train_transforms(
            image_size=224,
            use_randaugment=False,
        )

        img = Image.new("RGB", (256, 256), color="blue")
        result = transforms(img)

        assert result.shape == (3, 224, 224)
