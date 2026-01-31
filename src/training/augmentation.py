"""Data augmentation for training."""

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training transforms with standard augmentation.

    Args:
        image_size: Target image size.

    Returns:
        Compose of training transforms.
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
            transforms.RandomErasing(p=0.1),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Get validation transforms (no augmentation).

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
