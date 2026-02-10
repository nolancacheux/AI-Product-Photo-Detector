"""Dataset validation for AI image detection pipeline."""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
EXPECTED_CLASSES = {"real", "ai"}


def validate_dataset(data_dir: str) -> dict:
    """Validate dataset structure, integrity, and statistics.

    Args:
        data_dir: Path to the dataset root (expects train/test splits
                  with real/ and ai/ subdirectories).

    Returns:
        Dictionary containing the validation report.
    """
    data_path = Path(data_dir)
    report: dict = {
        "data_dir": str(data_path.resolve()),
        "valid": True,
        "errors": [],
        "warnings": [],
        "splits": {},
        "totals": {
            "images": 0,
            "corrupted": 0,
            "per_class": defaultdict(int),
        },
    }

    if not data_path.exists():
        report["valid"] = False
        report["errors"].append(f"Data directory does not exist: {data_dir}")
        return _finalize_report(report)

    splits = sorted(
        d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    if not splits:
        report["valid"] = False
        report["errors"].append("No split directories found (expected train/test)")
        return _finalize_report(report)

    for split in splits:
        split_path = data_path / split
        split_report = _validate_split(split_path)
        report["splits"][split] = split_report

        report["totals"]["images"] += split_report["total_images"]
        report["totals"]["corrupted"] += split_report["corrupted_count"]

        for cls, count in split_report["class_counts"].items():
            report["totals"]["per_class"][cls] += count

        if split_report["errors"]:
            report["errors"].extend(
                f"[{split}] {e}" for e in split_report["errors"]
            )

        if split_report["warnings"]:
            report["warnings"].extend(
                f"[{split}] {w}" for w in split_report["warnings"]
            )

    if report["errors"]:
        report["valid"] = False

    return _finalize_report(report)


def _validate_split(split_path: Path) -> dict:
    """Validate a single data split (train/test).

    Args:
        split_path: Path to the split directory.

    Returns:
        Split-level validation report.
    """
    split_report: dict = {
        "path": str(split_path),
        "class_counts": {},
        "total_images": 0,
        "corrupted_count": 0,
        "corrupted_files": [],
        "class_balance_ratio": None,
        "errors": [],
        "warnings": [],
        "resolution_stats": {
            "widths": [],
            "heights": [],
        },
        "file_size_stats": {
            "sizes_bytes": [],
        },
    }

    classes = sorted(
        d.name for d in split_path.iterdir() if d.is_dir()
    )

    if not classes:
        split_report["errors"].append("No class directories found")
        return _compute_stats(split_report)

    missing = EXPECTED_CLASSES - set(classes)
    if missing:
        split_report["errors"].append(
            f"Missing expected classes: {sorted(missing)}"
        )

    extra = set(classes) - EXPECTED_CLASSES
    if extra:
        split_report["warnings"].append(
            f"Unexpected class directories: {sorted(extra)}"
        )

    for cls in classes:
        cls_path = split_path / cls
        images = _list_images(cls_path)
        count = len(images)
        split_report["class_counts"][cls] = count
        split_report["total_images"] += count

        if count == 0:
            split_report["errors"].append(f"Class '{cls}' has no images")
            continue

        for img_path in images:
            try:
                file_size = img_path.stat().st_size
                split_report["file_size_stats"]["sizes_bytes"].append(file_size)

                with Image.open(img_path) as img:
                    img.verify()

                with Image.open(img_path) as img:
                    w, h = img.size
                    split_report["resolution_stats"]["widths"].append(w)
                    split_report["resolution_stats"]["heights"].append(h)
            except Exception as exc:
                split_report["corrupted_count"] += 1
                split_report["corrupted_files"].append(
                    {"file": str(img_path.relative_to(split_path)), "error": str(exc)}
                )

    counts = list(split_report["class_counts"].values())
    if len(counts) >= 2 and all(c > 0 for c in counts):
        ratio = min(counts) / max(counts)
        split_report["class_balance_ratio"] = round(ratio, 4)

        if ratio < 0.5:
            split_report["warnings"].append(
                f"Class imbalance detected: ratio={ratio:.4f} "
                f"(counts: {dict(split_report['class_counts'])})"
            )
    elif len(counts) == 1:
        split_report["warnings"].append("Only one class present")

    if split_report["corrupted_count"] > 0:
        pct = split_report["corrupted_count"] / max(split_report["total_images"], 1) * 100
        if pct > 5:
            split_report["errors"].append(
                f"High corruption rate: {split_report['corrupted_count']} "
                f"({pct:.1f}%) corrupted images"
            )
        else:
            split_report["warnings"].append(
                f"{split_report['corrupted_count']} corrupted images ({pct:.1f}%)"
            )

    return _compute_stats(split_report)


def _list_images(directory: Path) -> list[Path]:
    """List all valid image files in a directory.

    Args:
        directory: Path to scan for images.

    Returns:
        List of image file paths.
    """
    images = []
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            images.append(f)
    return sorted(images)


def _compute_stats(split_report: dict) -> dict:
    """Compute summary statistics for resolution and file sizes.

    Args:
        split_report: The split report to augment.

    Returns:
        Updated split report with computed stats.
    """
    widths = split_report["resolution_stats"]["widths"]
    heights = split_report["resolution_stats"]["heights"]
    sizes = split_report["file_size_stats"]["sizes_bytes"]

    summary: dict = {}

    if widths:
        summary["resolution"] = {
            "width": _summarize(widths),
            "height": _summarize(heights),
        }
    else:
        summary["resolution"] = None

    if sizes:
        summary["file_size_bytes"] = _summarize(sizes)
    else:
        summary["file_size_bytes"] = None

    # Replace raw lists with summary
    split_report["stats"] = summary
    del split_report["resolution_stats"]
    del split_report["file_size_stats"]

    return split_report


def _summarize(values: list) -> dict:
    """Compute min/max/mean/median for a list of numbers.

    Args:
        values: List of numeric values.

    Returns:
        Dictionary with summary statistics.
    """
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(sum(s) / n, 2),
        "median": median,
        "count": n,
    }


def _finalize_report(report: dict) -> dict:
    """Convert defaultdicts to regular dicts for JSON serialization.

    Args:
        report: The validation report.

    Returns:
        Cleaned report.
    """
    report["totals"]["per_class"] = dict(report["totals"]["per_class"])
    return report


def main() -> None:
    """CLI entrypoint for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate dataset for AI image detection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to dataset root directory (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any warnings are present",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Validating dataset at: %s", args.data_dir)
    report = validate_dataset(args.data_dir)

    report_json = json.dumps(report, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json)
        logger.info("Report saved to: %s", args.output)
    else:
        print(report_json)

    # Log summary
    logger.info(
        "Validation complete: valid=%s, images=%d, corrupted=%d, errors=%d, warnings=%d",
        report["valid"],
        report["totals"]["images"],
        report["totals"]["corrupted"],
        len(report["errors"]),
        len(report["warnings"]),
    )

    if not report["valid"]:
        logger.error("Dataset validation FAILED")
        sys.exit(1)
    elif args.strict and report["warnings"]:
        logger.warning("Dataset validation passed with warnings (strict mode)")
        sys.exit(1)
    else:
        logger.info("Dataset validation PASSED")


if __name__ == "__main__":
    main()
