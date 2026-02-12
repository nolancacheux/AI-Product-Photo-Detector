#!/usr/bin/env python3
"""Download and prepare a high-resolution real vs AI-generated product image dataset.

Strategies (in order of preference):
  1. HuggingFace CIFAKE upscaled to target resolution (default, no API key needed)
  2. HuggingFace Inference API generation for AI class + real product photos

For CIFAKE fallback at original 32x32 resolution, use scripts/download_cifake.py.

Output structure:
  data/processed/{train,val,test}/{real,ai_generated}/*.jpg
"""

import argparse
import io
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from scripts.download_utils import (
    LABEL_MAP,
    download_cifake,
    print_summary,
    split_and_save,
)

DEFAULT_TARGET_SIZE = 256
DEFAULT_MAX_PER_CLASS = 2500


def upscale_image(img: Image.Image, target_size: int) -> Image.Image:
    """Upscale an image to target_size x target_size using Lanczos resampling."""
    return img.convert("RGB").resize((target_size, target_size), Image.LANCZOS)


def generate_ai_images_hf(
    num_images: int,
    target_size: int,
    hf_token: str | None = None,
) -> list[Image.Image]:
    """Generate product images using HuggingFace Inference API.

    Requires a HuggingFace API token. Falls back gracefully if unavailable.
    """
    try:
        import requests
    except ImportError:
        print("  requests not available, skipping HF generation")
        return []

    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not hf_token:
        print("  No HF token found, skipping API generation")
        return []

    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}"}

    prompts = [
        "product photo of a leather handbag on white background, e-commerce, studio lighting",
        "product photo of wireless headphones on white background, e-commerce, professional",
        "product photo of a wristwatch on white background, e-commerce, studio lighting",
        "product photo of running shoes on white background, e-commerce, professional",
        "product photo of a coffee mug on white background, e-commerce, studio lighting",
        "product photo of sunglasses on white background, e-commerce, professional",
        "product photo of a laptop on white background, e-commerce, studio lighting",
        "product photo of a perfume bottle on white background, e-commerce, professional",
        "product photo of a backpack on white background, e-commerce, studio lighting",
        "product photo of a smartphone on white background, e-commerce, professional",
        "product photo of a ceramic vase on white background, e-commerce, studio lighting",
        "product photo of kitchen knife set on white background, e-commerce, professional",
        "product photo of a desk lamp on white background, e-commerce, studio lighting",
        "product photo of a water bottle on white background, e-commerce, professional",
        "product photo of a wallet on white background, e-commerce, studio lighting",
    ]

    images = []
    for i in tqdm(range(num_images), desc="  Generating via HF API"):
        prompt = prompts[i % len(prompts)]
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": prompt},
                timeout=120,
            )
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img = img.convert("RGB").resize(
                    (target_size, target_size), Image.LANCZOS,
                )
                images.append(img)
            else:
                print(f"  API error {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"  Generation failed: {e}")

    return images


def main() -> None:
    """Download dataset and prepare train/val/test splits."""
    parser = argparse.ArgumentParser(
        description="Download and prepare high-res real vs AI-generated dataset.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=DEFAULT_MAX_PER_CLASS,
        help=f"Max images per class (default: {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Target image size in pixels (default: {DEFAULT_TARGET_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["cifake-upscaled", "hf-api"],
        default="cifake-upscaled",
        help="Data source strategy (default: cifake-upscaled)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for image generation (hf-api mode)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"Output: {output_dir.resolve()}")
    print(f"Source: {args.source}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Max per class: {args.max_per_class}")
    print()

    if args.source == "cifake-upscaled":
        print("[1/3] Downloading CIFAKE dataset...")
        images_by_label = download_cifake(args.max_per_class)

        print(f"\n[2/3] Upscaling to {args.target_size}x{args.target_size}...")
        for label_id in images_by_label:
            label_name = LABEL_MAP[label_id]
            original = images_by_label[label_id]
            upscaled = [
                upscale_image(img, args.target_size)
                for img in tqdm(original, desc=f"  Upscaling {label_name}")
            ]
            images_by_label[label_id] = upscaled

    elif args.source == "hf-api":
        print("[1/3] Generating AI images via HuggingFace API...")
        ai_images = generate_ai_images_hf(
            args.max_per_class, args.target_size, args.hf_token,
        )
        if not ai_images:
            print("ERROR: No images generated. Check your HF token.", file=sys.stderr)
            sys.exit(1)

        print("[2/3] Downloading real images from CIFAKE...")
        cifake_data = download_cifake(args.max_per_class)
        real_images = [
            upscale_image(img, args.target_size) for img in cifake_data[0]
        ]

        images_by_label = {0: real_images, 1: ai_images}

    for label_id, label_name in LABEL_MAP.items():
        count = len(images_by_label[label_id])
        print(f"  {label_name}: {count} images at {args.target_size}x{args.target_size}")

    print(f"\n[3/3] Splitting and saving...")
    stats = split_and_save(
        images_by_label, output_dir,
        target_size=args.target_size, quality=92,
    )

    # Print summary
    print_summary(stats, output_dir)
    print(f"Image size: {args.target_size}x{args.target_size} JPG")


if __name__ == "__main__":
    main()
