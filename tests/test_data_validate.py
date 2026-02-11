"""Tests for dataset validation (src/data/validate.py)."""

from pathlib import Path

from PIL import Image

from src.data.validate import (
    _compute_stats,
    _finalize_report,
    _list_images,
    _summarize,
    _validate_split,
    validate_dataset,
)


def _create_test_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Create a valid JPEG test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, "red")
    img.save(str(path), format="JPEG")


class TestValidateDataset:
    """Tests for validate_dataset()."""

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        report = validate_dataset(str(tmp_path / "nonexistent"))
        assert report["valid"] is False
        assert any("does not exist" in e for e in report["errors"])

    def test_empty_directory(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        report = validate_dataset(str(data_dir))
        assert report["valid"] is False
        assert any("No split directories" in e for e in report["errors"])

    def test_valid_dataset(self, tmp_path: Path) -> None:
        # Create train/real and train/ai
        for cls in ("real", "ai"):
            for i in range(3):
                _create_test_image(tmp_path / "train" / cls / f"img_{i}.jpg")

        report = validate_dataset(str(tmp_path))
        assert report["valid"] is True
        assert report["totals"]["images"] == 6
        assert report["totals"]["corrupted"] == 0

    def test_missing_class_directory(self, tmp_path: Path) -> None:
        # Only "real" class, no "ai"
        for i in range(3):
            _create_test_image(tmp_path / "train" / "real" / f"img_{i}.jpg")

        report = validate_dataset(str(tmp_path))
        assert report["valid"] is False
        assert any("Missing expected classes" in e for e in report["errors"])

    def test_extra_class_directory(self, tmp_path: Path) -> None:
        for cls in ("real", "ai", "extra"):
            for i in range(3):
                _create_test_image(tmp_path / "train" / cls / f"img_{i}.jpg")

        report = validate_dataset(str(tmp_path))
        assert any("Unexpected class" in w for w in report["warnings"])

    def test_corrupted_image(self, tmp_path: Path) -> None:
        for cls in ("real", "ai"):
            for i in range(3):
                _create_test_image(tmp_path / "train" / cls / f"img_{i}.jpg")

        # Create a corrupted file
        corrupt_path = tmp_path / "train" / "real" / "corrupt.jpg"
        corrupt_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        report = validate_dataset(str(tmp_path))
        assert report["totals"]["corrupted"] > 0

    def test_multiple_splits(self, tmp_path: Path) -> None:
        for split in ("train", "test"):
            for cls in ("real", "ai"):
                for i in range(2):
                    _create_test_image(tmp_path / split / cls / f"img_{i}.jpg")

        report = validate_dataset(str(tmp_path))
        assert report["valid"] is True
        assert report["totals"]["images"] == 8
        assert "train" in report["splits"]
        assert "test" in report["splits"]


class TestValidateSplit:
    """Tests for _validate_split()."""

    def test_empty_split(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        report = _validate_split(split_dir)
        assert "No class directories" in report["errors"][0]

    def test_empty_class(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "real").mkdir(parents=True)
        (tmp_path / "train" / "ai").mkdir(parents=True)
        report = _validate_split(tmp_path / "train")
        assert any("has no images" in e for e in report["errors"])

    def test_class_imbalance_warning(self, tmp_path: Path) -> None:
        # 10 real, 2 ai -> ratio < 0.5
        for i in range(10):
            _create_test_image(tmp_path / "real" / f"img_{i}.jpg")
        for i in range(2):
            _create_test_image(tmp_path / "ai" / f"img_{i}.jpg")

        report = _validate_split(tmp_path)
        assert any("imbalance" in w.lower() for w in report["warnings"])

    def test_single_class_warning(self, tmp_path: Path) -> None:
        for i in range(3):
            _create_test_image(tmp_path / "real" / f"img_{i}.jpg")
        report = _validate_split(tmp_path)
        assert any("Only one class" in w for w in report["warnings"])


class TestListImages:
    """Tests for _list_images()."""

    def test_finds_valid_images(self, tmp_path: Path) -> None:
        _create_test_image(tmp_path / "img1.jpg")
        _create_test_image(tmp_path / "img2.png", size=(50, 50))
        # Create a non-image file
        (tmp_path / "readme.txt").write_text("not an image")

        images = _list_images(tmp_path)
        assert len(images) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        images = _list_images(tmp_path)
        assert images == []

    def test_results_sorted(self, tmp_path: Path) -> None:
        _create_test_image(tmp_path / "z.jpg")
        _create_test_image(tmp_path / "a.jpg")

        images = _list_images(tmp_path)
        assert images[0].name < images[1].name


class TestSummarize:
    """Tests for _summarize()."""

    def test_basic_stats(self) -> None:
        result = _summarize([1, 2, 3, 4, 5])
        assert result["min"] == 1
        assert result["max"] == 5
        assert result["mean"] == 3.0
        assert result["median"] == 3
        assert result["count"] == 5

    def test_even_count_median(self) -> None:
        result = _summarize([1, 2, 3, 4])
        assert result["median"] == 2.5

    def test_single_value(self) -> None:
        result = _summarize([42])
        assert result["min"] == 42
        assert result["max"] == 42
        assert result["mean"] == 42.0
        assert result["median"] == 42
        assert result["count"] == 1


class TestComputeStats:
    """Tests for _compute_stats()."""

    def test_with_data(self) -> None:
        report = {
            "resolution_stats": {"widths": [100, 200], "heights": [100, 200]},
            "file_size_stats": {"sizes_bytes": [1000, 2000]},
        }
        result = _compute_stats(report)
        assert "stats" in result
        assert result["stats"]["resolution"] is not None
        assert result["stats"]["file_size_bytes"] is not None

    def test_without_data(self) -> None:
        report = {
            "resolution_stats": {"widths": [], "heights": []},
            "file_size_stats": {"sizes_bytes": []},
        }
        result = _compute_stats(report)
        assert result["stats"]["resolution"] is None
        assert result["stats"]["file_size_bytes"] is None


class TestFinalizeReport:
    """Tests for _finalize_report()."""

    def test_converts_defaultdict(self) -> None:
        from collections import defaultdict

        report = {
            "totals": {
                "per_class": defaultdict(int, {"real": 5, "ai": 3}),
            },
        }
        result = _finalize_report(report)
        assert isinstance(result["totals"]["per_class"], dict)
        assert result["totals"]["per_class"]["real"] == 5
