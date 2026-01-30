"""Generate AI product images using Stable Diffusion.

This script generates synthetic product images for the AI class.
Requires: pip install diffusers transformers accelerate
"""

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Product categories and prompts
PRODUCT_CATEGORIES = [
    "electronics",
    "clothing",
    "furniture",
    "sports equipment",
    "kitchen appliances",
    "toys",
    "jewelry",
    "shoes",
    "bags",
    "watches",
]

PROMPT_TEMPLATES = [
    "professional product photo of {product} on white background, e-commerce style, studio lighting",
    "high quality product photography of {product}, isolated on white, commercial photography",
    "{product} product shot, clean white background, professional studio photo",
    "e-commerce listing photo of {product}, white backdrop, high resolution",
    "amazon style product image of {product}, pure white background, professional",
]

PRODUCT_EXAMPLES = {
    "electronics": [
        "wireless bluetooth headphones",
        "smartphone",
        "laptop computer",
        "tablet device",
        "smartwatch",
        "wireless earbuds",
        "portable speaker",
        "digital camera",
    ],
    "clothing": [
        "cotton t-shirt",
        "denim jeans",
        "leather jacket",
        "wool sweater",
        "summer dress",
        "business suit",
        "hoodie",
        "polo shirt",
    ],
    "furniture": [
        "modern office chair",
        "wooden dining table",
        "leather sofa",
        "bookshelf",
        "bed frame",
        "desk lamp",
        "coffee table",
        "wardrobe",
    ],
    "sports equipment": [
        "yoga mat",
        "dumbbell set",
        "tennis racket",
        "soccer ball",
        "running shoes",
        "bicycle helmet",
        "golf club",
        "basketball",
    ],
    "kitchen appliances": [
        "coffee maker",
        "blender",
        "toaster",
        "air fryer",
        "stand mixer",
        "electric kettle",
        "food processor",
        "microwave oven",
    ],
}


def generate_prompt() -> str:
    """Generate a random product prompt.
    
    Returns:
        Product photo prompt string.
    """
    category = random.choice(list(PRODUCT_EXAMPLES.keys()))
    product = random.choice(PRODUCT_EXAMPLES[category])
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(product=product)


def generate_images_diffusers(
    output_dir: Path,
    num_images: int = 100,
    model_id: str = "stabilityai/stable-diffusion-2-1-base",
    device: str = "auto",
) -> int:
    """Generate AI images using Hugging Face Diffusers.
    
    Args:
        output_dir: Output directory for images.
        num_images: Number of images to generate.
        model_id: Hugging Face model ID.
        device: Device to use (auto, cpu, cuda, mps).
        
    Returns:
        Number of images generated.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        logger.error("Please install diffusers: pip install diffusers transformers accelerate")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Loading model {model_id} on {device}")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    
    # Enable memory optimization
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    generated = 0
    for i in tqdm(range(num_images), desc="Generating AI images"):
        prompt = generate_prompt()
        
        try:
            image = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            
            # Resize to 224x224
            image = image.resize((224, 224))
            
            # Save
            filename = f"ai_generated_{i:04d}.jpg"
            image.save(output_dir / filename, "JPEG", quality=95)
            generated += 1
            
        except Exception as e:
            logger.warning(f"Failed to generate image {i}: {e}")
    
    logger.info(f"Generated {generated} AI images in {output_dir}")
    return generated


def generate_images_api(
    output_dir: Path,
    num_images: int = 100,
    api_url: str = "http://localhost:7860",
) -> int:
    """Generate AI images using local Stable Diffusion WebUI API.
    
    Args:
        output_dir: Output directory for images.
        num_images: Number of images to generate.
        api_url: Automatic1111 WebUI API URL.
        
    Returns:
        Number of images generated.
    """
    import base64
    import io
    
    import httpx
    from PIL import Image
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    for i in tqdm(range(num_images), desc="Generating via API"):
        prompt = generate_prompt()
        
        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "steps": 30,
            "width": 512,
            "height": 512,
            "cfg_scale": 7.5,
        }
        
        try:
            response = httpx.post(
                f"{api_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            
            data = response.json()
            image_b64 = data["images"][0]
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize to 224x224
            image = image.resize((224, 224))
            
            # Save
            filename = f"ai_generated_{i:04d}.jpg"
            image.save(output_dir / filename, "JPEG", quality=95)
            generated += 1
            
        except Exception as e:
            logger.warning(f"Failed to generate image {i}: {e}")
    
    logger.info(f"Generated {generated} AI images in {output_dir}")
    return generated


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate AI product images")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/ai_generated",
        help="Output directory",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--method",
        choices=["diffusers", "api"],
        default="diffusers",
        help="Generation method",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Model ID for diffusers method",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)",
    )
    args = parser.parse_args()
    
    setup_logging(level="INFO", json_format=False)
    output_dir = Path(args.output)
    
    if args.method == "diffusers":
        generate_images_diffusers(
            output_dir=output_dir,
            num_images=args.num_images,
            model_id=args.model,
            device=args.device,
        )
    else:
        generate_images_api(
            output_dir=output_dir,
            num_images=args.num_images,
        )


if __name__ == "__main__":
    main()
