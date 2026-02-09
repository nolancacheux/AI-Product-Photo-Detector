"""Tests for the dataset module."""

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.training.dataset import AIProductDataset, create_dataloaders


def _create_test_images(base_dir, real_count=3, ai_count=2):
    """Create test directory structure with dummy images."""
    real_dir = base_dir / "real"
    ai_dir = base_dir / "ai_generated"
    real_dir.mkdir(parents=True)
    ai_dir.mkdir(parents=True)

    for i in range(real_count):
        img = Image.new("RGB", (64, 64), color="green")
        img.save(real_dir / f"real_{i}.jpg")

    for i in range(ai_count):
        img = Image.new("RGB", (64, 64), color="red")
        img.save(ai_dir / f"ai_{i}.png")

    return base_dir


class TestAIProductDataset:
    """Tests for AIProductDataset."""

    def test_len(self, tmp_path) -> None:
        """Dataset length should match number of images."""
        _create_test_images(tmp_path, real_count=3, ai_count=2)
        dataset = AIProductDataset(data_dir=tmp_path, image_size=32)
        assert len(dataset) == 5

    def test_getitem_returns_tensor_and_label(self, tmp_path) -> None:
        """__getitem__ should return (tensor, int label)."""
        _create_test_images(tmp_path, real_count=1, ai_count=1)
        dataset = AIProductDataset(data_dir=tmp_path, image_size=32)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # RGB channels
        assert label in (0, 1)

    def test_labels_correct(self, tmp_path) -> None:
        """Real images should have label 0, AI images label 1."""
        _create_test_images(tmp_path, real_count=2, ai_count=3)
        dataset = AIProductDataset(data_dir=tmp_path, image_size=32)
        labels = [dataset[i][1] for i in range(len(dataset))]
        assert labels.count(0) == 2
        assert labels.count(1) == 3

    def test_empty_directory(self, tmp_path) -> None:
        """Dataset with no images should have length 0."""
        dataset = AIProductDataset(data_dir=tmp_path, image_size=32)
        assert len(dataset) == 0

    def test_custom_transform(self, tmp_path) -> None:
        """Custom transform should be applied."""
        from torchvision import transforms

        _create_test_images(tmp_path, real_count=1, ai_count=0)
        custom = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        dataset = AIProductDataset(data_dir=tmp_path, transform=custom, image_size=64)
        image, _ = dataset[0]
        assert image.shape == (3, 64, 64)


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_returns_two_dataloaders(self, tmp_path) -> None:
        """Should return a tuple of two DataLoader objects."""
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        _create_test_images(train_dir, real_count=4, ai_count=4)
        _create_test_images(val_dir, real_count=2, ai_count=2)

        train_loader, val_loader = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=2,
            num_workers=0,
            image_size=32,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_dataloader_batch_size(self, tmp_path) -> None:
        """DataLoader should yield batches of the specified size."""
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        _create_test_images(train_dir, real_count=4, ai_count=4)
        _create_test_images(val_dir, real_count=2, ai_count=2)

        train_loader, _ = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=4,
            num_workers=0,
            image_size=32,
        )
        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape[0] == 4
        assert labels.shape[0] == 4
