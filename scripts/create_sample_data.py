#!/usr/bin/env python3
"""Create sample dataset for testing.

This script creates synthetic colored images to test the training pipeline
without requiring real data or GPU for generation.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def create_real_style_image(size: int = 224) -> Image.Image:
    """Create an image that mimics real product photo characteristics.

    Real photos tend to have:
    - Natural, muted colors
    - Gradients and shadows
    - Some noise/texture
    """
    # Natural color palette
    colors = [
        (139, 119, 101),  # Warm gray
        (180, 160, 140),  # Beige
        (100, 120, 100),  # Olive
        (150, 140, 130),  # Taupe
        (120, 110, 100),  # Brown gray
    ]

    base_color = random.choice(colors)
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)

    # Add some rectangles to simulate product shapes
    for _ in range(random.randint(1, 3)):
        x1 = random.randint(20, size // 2)
        y1 = random.randint(20, size // 2)
        x2 = x1 + random.randint(40, size // 2)
        y2 = y1 + random.randint(40, size // 2)

        # Slightly different shade
        rect_color = tuple(min(255, max(0, c + random.randint(-30, 30))) for c in base_color)
        draw.rectangle([x1, y1, x2, y2], fill=rect_color)

    # Add slight blur to simulate photo softness
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Add noise
    pixels = img.load()
    for i in range(size):
        for j in range(size):
            r, g, b = pixels[i, j]
            noise = random.randint(-5, 5)
            pixels[i, j] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )

    return img


def create_ai_style_image(size: int = 224) -> Image.Image:
    """Create an image that mimics AI-generated characteristics.

    AI images tend to have:
    - Vibrant, saturated colors
    - Perfect gradients
    - Unnatural perfection
    - Sometimes artifacts
    """
    # Vibrant color palette
    colors = [
        (255, 100, 150),  # Pink
        (100, 200, 255),  # Cyan
        (255, 180, 50),  # Orange
        (150, 100, 255),  # Purple
        (100, 255, 150),  # Lime
    ]

    base_color = random.choice(colors)
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)

    # Add perfect geometric shapes (AI tends to be "too perfect")
    for _ in range(random.randint(2, 5)):
        shape_type = random.choice(["rectangle", "ellipse"])
        x1 = random.randint(10, size - 60)
        y1 = random.randint(10, size - 60)
        x2 = x1 + random.randint(30, 80)
        y2 = y1 + random.randint(30, 80)

        # Contrasting bright color
        shape_color = tuple(min(255, max(0, 255 - c + random.randint(-20, 20))) for c in base_color)

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=shape_color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=shape_color)

    # Add gradient effect (common in AI images)
    for y in range(size):
        alpha = y / size
        for x in range(size):
            r, g, b = img.getpixel((x, y))
            # Slight color shift
            r = int(r * (1 - alpha * 0.1))
            b = int(b * (1 + alpha * 0.1))
            img.putpixel((x, y), (min(255, r), g, min(255, b)))

    return img


def create_sample_dataset(
    output_dir: Path,
    train_samples: int = 200,
    val_samples: int = 50,
    test_samples: int = 50,
) -> None:
    """Create a complete sample dataset.

    Args:
        output_dir: Output directory for dataset.
        train_samples: Number of training samples per class.
        val_samples: Number of validation samples per class.
        test_samples: Number of test samples per class.
    """
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split_name, num_samples in splits.items():
        # Create real images
        real_dir = output_dir / split_name / "real"
        real_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating {num_samples} real images for {split_name}...")
        for i in range(num_samples):
            img = create_real_style_image()
            img.save(real_dir / f"real_{i:04d}.jpg", "JPEG", quality=90)

        # Create AI images
        ai_dir = output_dir / split_name / "ai_generated"
        ai_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating {num_samples} AI images for {split_name}...")
        for i in range(num_samples):
            img = create_ai_style_image()
            img.save(ai_dir / f"ai_{i:04d}.jpg", "JPEG", quality=90)

    print(f"\nDataset created at {output_dir}")
    print(f"  Train: {train_samples * 2} images")
    print(f"  Val: {val_samples * 2} images")
    print(f"  Test: {test_samples * 2} images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create sample dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)

    args = parser.parse_args()

    create_sample_dataset(
        output_dir=Path(args.output),
        train_samples=args.train,
        val_samples=args.val,
        test_samples=args.test,
    )
