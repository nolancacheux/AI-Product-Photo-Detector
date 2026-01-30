"""Dataset and data loading utilities."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class AIProductDataset(Dataset):
    """Dataset for AI vs Real product image classification.

    Expected directory structure:
        data_dir/
        ├── real/
        │   ├── image1.jpg
        │   └── ...
        └── ai_generated/
            ├── image1.jpg
            └── ...

    Labels:
        - 0: Real image
        - 1: AI-generated image
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: transforms.Compose | None = None,
        image_size: int = 224,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Path to data directory.
            transform: Optional custom transforms.
            image_size: Target image size.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        # Default transform if none provided
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

        # Collect image paths and labels
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms.

        Returns:
            Compose of default transforms.
        """
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_samples(self) -> None:
        """Load image paths and labels from directory."""
        # Real images (label = 0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((img_path, 0))

        # AI-generated images (label = 1)
        ai_dir = self.data_dir / "ai_generated"
        if ai_dir.exists():
            for img_path in ai_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((img_path, 1))

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image tensor, label).
        """
        img_path, label = self.samples[idx]

        # Load and convert image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training data augmentation transforms.

    Args:
        image_size: Target image size.

    Returns:
        Compose of training transforms with augmentation.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Compose of validation transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Args:
        train_dir: Path to training data.
        val_dir: Path to validation data.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        image_size: Target image size.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = AIProductDataset(
        data_dir=train_dir,
        transform=get_train_transforms(image_size),
        image_size=image_size,
    )

    val_dataset = AIProductDataset(
        data_dir=val_dir,
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
