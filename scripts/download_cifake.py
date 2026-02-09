#!/usr/bin/env python3
"""Download and prepare the CIFAKE dataset for training.

CIFAKE contains real CIFAR-10 images and AI-generated images from Stable Diffusion.
Images are split into train/val/test sets and saved to the project data directory.

Source: https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset
"""

import argparse
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


def download_dataset(max_per_class: int) -> dict:
    """Download the CIFAKE dataset from HuggingFace.

    Tries multiple dataset identifiers in case the primary one fails.

    Args:
        max_per_class: Maximum number of images per class to download.

    Returns:
        Dictionary mapping label (0 or 1) to list of PIL Images.
    """
    from datasets import load_dataset

    dataset = None
    used_name = None

    for name in DATASET_CANDIDATES:
        try:
            print(f"Attempting to load dataset: {name}")
            dataset = load_dataset(name, split="train", trust_remote_code=True)
            used_name = name
            print(f"Successfully loaded: {name}")
            break
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue

    if dataset is None:
        print("ERROR: Could not load any CIFAKE dataset variant.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {used_name}")
    print(f"Total samples available: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    images_by_label: dict[int, list[Image.Image]] = {0: [], 1: []}

    for sample in tqdm(dataset, desc="Loading images"):
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

    splits = {
        "train": [images[i] for i in indices[:n_train]],
        "val": [images[i] for i in indices[n_train : n_train + n_val]],
        "test": [images[i] for i in indices[n_train + n_val :]],
    }

    return splits


def save_images(
    images: list[Image.Image],
    output_dir: Path,
    prefix: str,
) -> int:
    """Save a list of PIL Images as JPEG files.

    Args:
        images: List of PIL Images to save.
        output_dir: Directory to save images in.
        prefix: Filename prefix (e.g. 'real' or 'ai').

    Returns:
        Number of images saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for i, img in enumerate(images):
        img_rgb = img.convert("RGB")
        img_rgb.save(output_dir / f"{prefix}_{i:05d}.jpg", "JPEG", quality=90)
        count += 1

    return count


def main() -> None:
    """Download CIFAKE dataset and save to project data directory."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the CIFAKE dataset.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2500,
        help="Maximum images per class (default: 2500, total will be 2x this)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Max per class: {args.max_per_class}")
    print()

    # Download
    images_by_label = download_dataset(args.max_per_class)

    for label_id, label_name in LABEL_MAP.items():
        print(f"  {label_name}: {len(images_by_label[label_id])} images loaded")

    # Split
    rng = np.random.default_rng(SEED)
    stats: dict[str, dict[str, int]] = {}

    for label_id, label_name in LABEL_MAP.items():
        splits = split_images(images_by_label[label_id], rng)

        for split_name, split_images_list in splits.items():
            dest = output_dir / split_name / label_name
            count = save_images(split_images_list, dest, label_name)

            if split_name not in stats:
                stats[split_name] = {}
            stats[split_name][label_name] = count

    # Print summary
    print()
    print("=" * 50)
    print("Dataset prepared successfully")
    print("=" * 50)

    total = 0
    for split_name in ["train", "val", "test"]:
        split_total = sum(stats[split_name].values())
        total += split_total
        print(f"  {split_name:6s}: {split_total:5d} images", end="  (")
        parts = [f"{k}: {v}" for k, v in stats[split_name].items()]
        print(", ".join(parts) + ")")

    print(f"  {'total':6s}: {total:5d} images")
    print()
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
