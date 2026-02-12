#!/usr/bin/env python3
"""Download and prepare the CIFAKE dataset for training.

CIFAKE contains real CIFAR-10 images and AI-generated images from Stable Diffusion.
Images are split into train/val/test sets and saved to the project data directory.

Source: https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset
"""

import argparse
from pathlib import Path

from scripts.download_utils import (
    LABEL_MAP,
    download_cifake,
    print_summary,
    split_and_save,
)


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
    images_by_label = download_cifake(args.max_per_class)

    for label_id, label_name in LABEL_MAP.items():
        print(f"  {label_name}: {len(images_by_label[label_id])} images loaded")

    # Split and save
    stats = split_and_save(images_by_label, output_dir)

    # Print summary
    print_summary(stats, output_dir)


if __name__ == "__main__":
    main()
