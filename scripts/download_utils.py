"""Shared utilities for dataset download scripts.

Extracted from download_cifake.py and download_dataset.py to eliminate
code duplication across download scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_CANDIDATES = [
    "dragonintelligence/CIFAKE-image-dataset",
    "yanbax/CIFAKE_autotrain_compatible",
]

LABEL_MAP = {0: "real", 1: "ai_generated"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def download_cifake(max_per_class: int) -> dict[int, list[Image.Image]]:
    """Download the CIFAKE dataset from HuggingFace.

    Tries multiple dataset identifiers in case the primary one fails.

    Args:
        max_per_class: Maximum number of images per class to download.

    Returns:
        Dictionary mapping label (0=real, 1=ai_generated) to list of PIL Images.
    """
    from datasets import load_dataset

    dataset = None
    used_name = None

    for name in DATASET_CANDIDATES:
        try:
            print(f"  Trying: {name}")
            dataset = load_dataset(name, split="train", trust_remote_code=True)
            used_name = name
            print(f"  Loaded: {name} ({len(dataset)} samples)")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if dataset is None:
        print("ERROR: Could not load any CIFAKE dataset variant.", file=sys.stderr)
        sys.exit(1)

    print(f"  Source: {used_name}")

    images_by_label: dict[int, list[Image.Image]] = {0: [], 1: []}

    for sample in tqdm(dataset, desc="  Loading images"):
        label = int(sample["label"])
        if label not in images_by_label:
            continue
        if len(images_by_label[label]) >= max_per_class:
            if all(len(v) >= max_per_class for v in images_by_label.values()):
                break
            continue
        images_by_label[label].append(sample["image"])

    return images_by_label


def split_images(
    images: list[Image.Image],
    rng: np.random.Generator,
) -> dict[str, list[Image.Image]]:
    """Split a list of images into train/val/test sets.

    Args:
        images: List of PIL Images.
        rng: Numpy random generator for reproducibility.

    Returns:
        Dictionary mapping split name to list of PIL Images.
    """
    indices = np.arange(len(images))
    rng.shuffle(indices)

    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": [images[i] for i in indices[:n_train]],
        "val": [images[i] for i in indices[n_train : n_train + n_val]],
        "test": [images[i] for i in indices[n_train + n_val :]],
    }


def save_images(
    images: list[Image.Image],
    output_dir: Path,
    prefix: str,
    target_size: int | None = None,
    quality: int = 90,
) -> int:
    """Save a list of PIL Images as JPEG files.

    Args:
        images: List of PIL Images to save.
        output_dir: Directory to save images in.
        prefix: Filename prefix (e.g. 'real' or 'ai').
        target_size: If set, resize images to this square size.
        quality: JPEG quality (1-100).

    Returns:
        Number of images saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for i, img in enumerate(images):
        img_rgb = img.convert("RGB")
        if target_size and img_rgb.size != (target_size, target_size):
            img_rgb = img_rgb.resize((target_size, target_size), Image.LANCZOS)
        img_rgb.save(output_dir / f"{prefix}_{i:05d}.jpg", "JPEG", quality=quality)
        count += 1

    return count


def print_summary(stats: dict[str, dict[str, int]], output_dir: Path) -> None:
    """Print a formatted summary of the prepared dataset.

    Args:
        stats: Nested dict of split_name -> class_name -> count.
        output_dir: Where the dataset was saved.
    """
    print()
    print("=" * 55)
    print("Dataset prepared successfully")
    print("=" * 55)

    total = 0
    for split_name in ["train", "val", "test"]:
        split_total = sum(stats[split_name].values())
        total += split_total
        parts = [f"{k}: {v}" for k, v in stats[split_name].items()]
        print(f"  {split_name:6s}: {split_total:5d} images  ({', '.join(parts)})")

    print(f"  {'total':6s}: {total:5d} images")
    print(f"\nSaved to: {output_dir.resolve()}")


def split_and_save(
    images_by_label: dict[int, list[Image.Image]],
    output_dir: Path,
    target_size: int | None = None,
    quality: int = 90,
) -> dict[str, dict[str, int]]:
    """Split images by label into train/val/test and save them.

    Args:
        images_by_label: Dict mapping label int to list of PIL Images.
        output_dir: Base output directory.
        target_size: Optional square resize target.
        quality: JPEG quality.

    Returns:
        Stats dict: split_name -> class_name -> count.
    """
    rng = np.random.default_rng(SEED)
    stats: dict[str, dict[str, int]] = {}

    for label_id, label_name in LABEL_MAP.items():
        splits = split_images(images_by_label[label_id], rng)

        for split_name, split_images_list in splits.items():
            dest = output_dir / split_name / label_name
            count = save_images(
                split_images_list, dest, label_name,
                target_size=target_size, quality=quality,
            )
            if split_name not in stats:
                stats[split_name] = {}
            stats[split_name][label_name] = count

    return stats
