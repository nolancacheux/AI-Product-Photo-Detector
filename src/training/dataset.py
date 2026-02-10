"""Dataset and data loading utilities."""

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.training.augmentation import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


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

        if len(self.samples) == 0:
            logger.warning(f"No samples found in {self.data_dir}. Check directory structure.")

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

        try:
            # Load and convert image
            image = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label
        except (OSError, IOError) as e:
            logger.warning(f"Failed to load image {img_path}: {e}, returning random replacement")
            # Return a random valid sample instead of crashing
            replacement_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(replacement_idx)


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

    use_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )

    return train_loader, val_loader
