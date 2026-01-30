"""Dataset download utilities."""

import hashlib
import shutil
from pathlib import Path

import httpx
from rich.progress import Progress, TaskID
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Real product image sources (free datasets)
DATASET_SOURCES = {
    "stanford_products": {
        "url": "https://huggingface.co/datasets/Marqo/Stanford-Online-Products/resolve/main/data/images.tar.gz",
        "description": "Stanford Online Products - Real e-commerce images",
    },
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from.
        dest_path: Destination file path.
        chunk_size: Download chunk size.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a tar.gz archive.
    
    Args:
        archive_path: Path to archive file.
        dest_dir: Destination directory.
    """
    import tarfile
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {archive_path} to {dest_dir}")
    
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def prepare_real_images(
    source_dir: Path,
    dest_dir: Path,
    max_images: int = 2500,
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
) -> int:
    """Prepare real product images from source directory.
    
    Args:
        source_dir: Source directory with images.
        dest_dir: Destination directory.
        max_images: Maximum number of images to copy.
        image_extensions: Valid image extensions.
        
    Returns:
        Number of images copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    images = []
    for ext in image_extensions:
        images.extend(source_dir.rglob(f"*{ext}"))
        images.extend(source_dir.rglob(f"*{ext.upper()}"))
    
    # Limit and copy
    copied = 0
    for img_path in tqdm(images[:max_images], desc="Copying real images"):
        # Generate unique filename
        file_hash = hashlib.md5(str(img_path).encode()).hexdigest()[:8]
        new_name = f"real_{file_hash}{img_path.suffix.lower()}"
        dest_path = dest_dir / new_name
        
        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
            copied += 1
    
    logger.info(f"Copied {copied} real images to {dest_dir}")
    return copied


def create_sample_dataset(data_dir: Path, num_samples: int = 100) -> None:
    """Create a small sample dataset for testing.
    
    Creates synthetic images (colored squares) for testing purposes.
    
    Args:
        data_dir: Data directory.
        num_samples: Number of samples per class.
    """
    from PIL import Image
    import random
    
    real_dir = data_dir / "real"
    ai_dir = data_dir / "ai_generated"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)
    
    # Create "real" images (natural colors)
    natural_colors = [
        (139, 69, 19),   # Brown
        (34, 139, 34),   # Forest Green
        (70, 130, 180),  # Steel Blue
        (210, 180, 140), # Tan
        (128, 128, 128), # Gray
    ]
    
    for i in range(num_samples):
        color = random.choice(natural_colors)
        # Add some noise to make it look more natural
        color = tuple(min(255, max(0, c + random.randint(-20, 20))) for c in color)
        img = Image.new("RGB", (224, 224), color)
        img.save(real_dir / f"real_sample_{i:04d}.jpg", "JPEG")
    
    # Create "AI" images (vibrant/synthetic colors)
    ai_colors = [
        (255, 0, 128),   # Hot Pink
        (0, 255, 255),   # Cyan
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (0, 255, 0),     # Lime
    ]
    
    for i in range(num_samples):
        color = random.choice(ai_colors)
        img = Image.new("RGB", (224, 224), color)
        img.save(ai_dir / f"ai_sample_{i:04d}.jpg", "JPEG")
    
    logger.info(f"Created sample dataset: {num_samples} real, {num_samples} AI images")


if __name__ == "__main__":
    # Create sample dataset for testing
    create_sample_dataset(Path("data/processed/train"), num_samples=80)
    create_sample_dataset(Path("data/processed/val"), num_samples=20)
