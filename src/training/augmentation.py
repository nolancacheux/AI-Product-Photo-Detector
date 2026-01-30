"""Advanced data augmentation techniques."""

import random
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms


class CutMix:
    """CutMix augmentation for binary classification.

    Randomly cuts a patch from one image and pastes it onto another,
    mixing labels proportionally to the area.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5) -> None:
        """Initialize CutMix.

        Args:
            alpha: Beta distribution parameter for lambda sampling.
            prob: Probability of applying CutMix.
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix to a batch.

        Args:
            images: Batch of images (N, C, H, W).
            labels: Batch of labels (N,).

        Returns:
            Tuple of (mixed images, mixed labels).
        """
        if random.random() > self.prob:
            return images, labels.float()

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for mixing
        rand_index = torch.randperm(batch_size)

        # Get bounding box
        _, _, h, w = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]

        # Adjust labels
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        labels = lam * labels.float() + (1 - lam) * labels[rand_index].float()

        return images, labels


class MixUp:
    """MixUp augmentation for binary classification.

    Linearly interpolates between two images and their labels.
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5) -> None:
        """Initialize MixUp.

        Args:
            alpha: Beta distribution parameter for lambda sampling.
            prob: Probability of applying MixUp.
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp to a batch.

        Args:
            images: Batch of images (N, C, H, W).
            labels: Batch of labels (N,).

        Returns:
            Tuple of (mixed images, mixed labels).
        """
        if random.random() > self.prob:
            return images, labels.float()

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for mixing
        rand_index = torch.randperm(batch_size)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[rand_index]

        # Mix labels
        mixed_labels = lam * labels.float() + (1 - lam) * labels[rand_index].float()

        return mixed_images, mixed_labels


class RandAugment:
    """RandAugment: randomly applies N augmentations with magnitude M."""

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        fill_color: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        """Initialize RandAugment.

        Args:
            num_ops: Number of augmentation operations to apply.
            magnitude: Magnitude of augmentation (0-30).
            fill_color: Fill color for affine transforms.
        """
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.fill_color = fill_color

        # Define augmentation operations
        self.augmentations = [
            self._auto_contrast,
            self._equalize,
            self._invert,
            self._rotate,
            self._posterize,
            self._solarize,
            self._color,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

    def _magnitude_to_value(self, magnitude: int, max_val: float) -> float:
        """Convert magnitude to actual value."""
        return (magnitude / 30.0) * max_val

    def _auto_contrast(self, img: Image.Image, magnitude: int) -> Image.Image:
        return ImageOps.autocontrast(img)

    def _equalize(self, img: Image.Image, magnitude: int) -> Image.Image:
        return ImageOps.equalize(img)

    def _invert(self, img: Image.Image, magnitude: int) -> Image.Image:
        return ImageOps.invert(img)

    def _rotate(self, img: Image.Image, magnitude: int) -> Image.Image:
        degrees = self._magnitude_to_value(magnitude, 30)
        if random.random() > 0.5:
            degrees = -degrees
        return img.rotate(degrees, fillcolor=self.fill_color)

    def _posterize(self, img: Image.Image, magnitude: int) -> Image.Image:
        bits = int(8 - self._magnitude_to_value(magnitude, 4))
        bits = max(1, bits)
        return ImageOps.posterize(img, bits)

    def _solarize(self, img: Image.Image, magnitude: int) -> Image.Image:
        threshold = int(256 - self._magnitude_to_value(magnitude, 256))
        return ImageOps.solarize(img, threshold)

    def _color(self, img: Image.Image, magnitude: int) -> Image.Image:
        factor = 1.0 + self._magnitude_to_value(magnitude, 0.9)
        if random.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Color(img).enhance(factor)

    def _contrast(self, img: Image.Image, magnitude: int) -> Image.Image:
        factor = 1.0 + self._magnitude_to_value(magnitude, 0.9)
        if random.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Contrast(img).enhance(factor)

    def _brightness(self, img: Image.Image, magnitude: int) -> Image.Image:
        factor = 1.0 + self._magnitude_to_value(magnitude, 0.9)
        if random.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Brightness(img).enhance(factor)

    def _sharpness(self, img: Image.Image, magnitude: int) -> Image.Image:
        factor = 1.0 + self._magnitude_to_value(magnitude, 0.9)
        if random.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Sharpness(img).enhance(factor)

    def _shear_x(self, img: Image.Image, magnitude: int) -> Image.Image:
        shear = self._magnitude_to_value(magnitude, 0.3)
        if random.random() > 0.5:
            shear = -shear
        return img.transform(
            img.size,
            Image.Transform.AFFINE,
            (1, shear, 0, 0, 1, 0),
            fillcolor=self.fill_color,
        )

    def _shear_y(self, img: Image.Image, magnitude: int) -> Image.Image:
        shear = self._magnitude_to_value(magnitude, 0.3)
        if random.random() > 0.5:
            shear = -shear
        return img.transform(
            img.size,
            Image.Transform.AFFINE,
            (1, 0, 0, shear, 1, 0),
            fillcolor=self.fill_color,
        )

    def _translate_x(self, img: Image.Image, magnitude: int) -> Image.Image:
        pixels = int(self._magnitude_to_value(magnitude, img.width * 0.3))
        if random.random() > 0.5:
            pixels = -pixels
        return img.transform(
            img.size,
            Image.Transform.AFFINE,
            (1, 0, pixels, 0, 1, 0),
            fillcolor=self.fill_color,
        )

    def _translate_y(self, img: Image.Image, magnitude: int) -> Image.Image:
        pixels = int(self._magnitude_to_value(magnitude, img.height * 0.3))
        if random.random() > 0.5:
            pixels = -pixels
        return img.transform(
            img.size,
            Image.Transform.AFFINE,
            (1, 0, 0, 0, 1, pixels),
            fillcolor=self.fill_color,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations.

        Args:
            img: Input PIL image.

        Returns:
            Augmented PIL image.
        """
        ops = random.choices(self.augmentations, k=self.num_ops)
        for op in ops:
            img = op(img, self.magnitude)
        return img


class GridMask:
    """GridMask augmentation - drops out rectangular regions in a grid pattern."""

    def __init__(
        self,
        d1: int = 96,
        d2: int = 224,
        rotate: float = 1.0,
        ratio: float = 0.5,
        prob: float = 0.5,
    ) -> None:
        """Initialize GridMask.

        Args:
            d1: Minimum grid size.
            d2: Maximum grid size.
            rotate: Rotation range in radians.
            ratio: Ratio of masked region to grid cell.
            prob: Probability of applying GridMask.
        """
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply GridMask to image tensor.

        Args:
            img: Input tensor (C, H, W).

        Returns:
            Masked tensor.
        """
        if random.random() > self.prob:
            return img

        _, h, w = img.shape
        d = random.randint(self.d1, self.d2)
        l = int(d * self.ratio)

        mask = torch.ones(h, w)
        for i in range(-d, h, d):
            for j in range(-d, w, d):
                x1, y1 = max(0, i), max(0, j)
                x2, y2 = min(h, i + l), min(w, j + l)
                mask[x1:x2, y1:y2] = 0

        return img * mask.unsqueeze(0)


def get_advanced_train_transforms(
    image_size: int = 224,
    use_randaugment: bool = True,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
) -> transforms.Compose:
    """Get advanced training transforms with RandAugment.

    Args:
        image_size: Target image size.
        use_randaugment: Whether to use RandAugment.
        randaugment_n: Number of RandAugment operations.
        randaugment_m: RandAugment magnitude.

    Returns:
        Compose of training transforms.
    """
    transform_list = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if use_randaugment:
        transform_list.append(RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

    return transforms.Compose(transform_list)


class MixUpCutMixCollator:
    """Collator that applies MixUp or CutMix to batches."""

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        switch_prob: float = 0.5,
    ) -> None:
        """Initialize collator.

        Args:
            mixup_alpha: MixUp alpha parameter.
            cutmix_alpha: CutMix alpha parameter.
            mixup_prob: MixUp probability.
            cutmix_prob: CutMix probability.
            switch_prob: Probability of using CutMix vs MixUp.
        """
        self.mixup = MixUp(alpha=mixup_alpha, prob=mixup_prob)
        self.cutmix = CutMix(alpha=cutmix_alpha, prob=cutmix_prob)
        self.switch_prob = switch_prob

    def __call__(
        self,
        batch: list[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate batch and apply augmentation.

        Args:
            batch: List of (image, label) tuples.

        Returns:
            Tuple of (images, labels) tensors.
        """
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])

        if random.random() < self.switch_prob:
            images, labels = self.cutmix(images, labels)
        else:
            images, labels = self.mixup(images, labels)

        return images, labels
