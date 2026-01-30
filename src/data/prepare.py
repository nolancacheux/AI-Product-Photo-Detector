"""Data preparation and splitting utilities."""

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def split_dataset(
    real_dir: Path,
    ai_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split dataset into train/val/test sets.
    
    Args:
        real_dir: Directory with real images.
        ai_dir: Directory with AI-generated images.
        output_dir: Output directory for splits.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with split statistics.
    """
    random.seed(seed)
    
    # Verify ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    # Collect images
    real_images = list(real_dir.glob("*.[jJ][pP][gG]")) + \
                  list(real_dir.glob("*.[jJ][pP][eE][gG]")) + \
                  list(real_dir.glob("*.[pP][nN][gG]"))
    
    ai_images = list(ai_dir.glob("*.[jJ][pP][gG]")) + \
                list(ai_dir.glob("*.[jJ][pP][eE][gG]")) + \
                list(ai_dir.glob("*.[pP][nN][gG]"))
    
    logger.info(f"Found {len(real_images)} real images, {len(ai_images)} AI images")
    
    # Shuffle
    random.shuffle(real_images)
    random.shuffle(ai_images)
    
    # Calculate split sizes
    def split_list(items: list, train_r: float, val_r: float) -> tuple:
        n = len(items)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return items[:train_end], items[train_end:val_end], items[val_end:]
    
    real_train, real_val, real_test = split_list(real_images, train_ratio, val_ratio)
    ai_train, ai_val, ai_test = split_list(ai_images, train_ratio, val_ratio)
    
    # Create output directories
    splits = {
        "train": {"real": real_train, "ai_generated": ai_train},
        "val": {"real": real_val, "ai_generated": ai_val},
        "test": {"real": real_test, "ai_generated": ai_test},
    }
    
    stats = {}
    
    for split_name, classes in splits.items():
        for class_name, images in classes.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(images, desc=f"{split_name}/{class_name}"):
                shutil.copy2(img_path, dest_dir / img_path.name)
            
            stats[f"{split_name}_{class_name}"] = len(images)
    
    # Log statistics
    logger.info("Dataset split complete:")
    logger.info(f"  Train: {len(real_train)} real, {len(ai_train)} AI")
    logger.info(f"  Val:   {len(real_val)} real, {len(ai_val)} AI")
    logger.info(f"  Test:  {len(real_test)} real, {len(ai_test)} AI")
    
    return stats


def validate_dataset(data_dir: Path) -> dict:
    """Validate dataset structure and contents.
    
    Args:
        data_dir: Data directory to validate.
        
    Returns:
        Validation results.
    """
    from PIL import Image
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    expected_structure = ["train/real", "train/ai_generated", "val/real", "val/ai_generated"]
    
    for subdir in expected_structure:
        dir_path = data_dir / subdir
        
        if not dir_path.exists():
            results["errors"].append(f"Missing directory: {subdir}")
            results["valid"] = False
            continue
        
        # Count and validate images
        images = list(dir_path.glob("*"))
        valid_images = 0
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images += 1
            except Exception as e:
                results["warnings"].append(f"Invalid image {img_path}: {e}")
        
        results["stats"][subdir] = {
            "total": len(images),
            "valid": valid_images,
        }
        
        if valid_images < 10:
            results["warnings"].append(f"Very few images in {subdir}: {valid_images}")
    
    # Check class balance
    for split in ["train", "val"]:
        real_count = results["stats"].get(f"{split}/real", {}).get("valid", 0)
        ai_count = results["stats"].get(f"{split}/ai_generated", {}).get("valid", 0)
        
        if real_count > 0 and ai_count > 0:
            ratio = max(real_count, ai_count) / min(real_count, ai_count)
            if ratio > 2:
                results["warnings"].append(
                    f"Class imbalance in {split}: {real_count} real vs {ai_count} AI"
                )
    
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prepare dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--real-dir", type=str, required=True)
    split_parser.add_argument("--ai-dir", type=str, required=True)
    split_parser.add_argument("--output-dir", type=str, default="data/processed")
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--val-ratio", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=42)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--data-dir", type=str, default="data/processed")
    
    args = parser.parse_args()
    setup_logging(level="INFO", json_format=False)
    
    if args.command == "split":
        split_dataset(
            real_dir=Path(args.real_dir),
            ai_dir=Path(args.ai_dir),
            output_dir=Path(args.output_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1 - args.train_ratio - args.val_ratio,
            seed=args.seed,
        )
    elif args.command == "validate":
        results = validate_dataset(Path(args.data_dir))
        
        print("\n=== Dataset Validation ===")
        print(f"Valid: {'✅' if results['valid'] else '❌'}")
        
        if results["errors"]:
            print("\nErrors:")
            for err in results["errors"]:
                print(f"  ❌ {err}")
        
        if results["warnings"]:
            print("\nWarnings:")
            for warn in results["warnings"]:
                print(f"  ⚠️ {warn}")
        
        print("\nStatistics:")
        for path, stats in results["stats"].items():
            print(f"  {path}: {stats['valid']}/{stats['total']} valid")


if __name__ == "__main__":
    main()
