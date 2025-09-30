#!/usr/bin/env python3
"""
Generate 7,000-10,000 Monet-style images in JPG format for submission.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

from generator import Generator
from configs import AppParams
import device


def denormalize_image(tensor):
    """Convert from [-1, 1] to [0, 255] uint8"""
    # Convert from [-1, 1] to [0, 1]
    denorm = (tensor + 1.0) / 2.0
    denorm = torch.clamp(denorm, 0.0, 1.0)
    # Convert to [0, 255] uint8
    return (denorm * 255).to(torch.uint8)


def generate_submission_images(generator_path: str, output_dir: str, num_images: int = 8000, batch_size: int = 64, latent_size: int = 128):
    """
    Generate Monet-style images for submission.

    Args:
        generator_path: Path to saved generator model
        output_dir: Directory to save generated images
        num_images: Number of images to generate (7000-10000)
        batch_size: Batch size for generation
        latent_size: Size of latent vector (default: 128)
    """

    # Setup device
    device_info = device.get_device_info()
    print(f"Using device: {device_info.backend}")

    # Load generator with specified latent_size
    config = AppParams()

    # Check if generator file exists
    if not os.path.exists(generator_path):
        print(f"Generator not found at {generator_path}")
        print("Train the model first using: uv run python main.py gan --epochs 500")
        return

    print(f"Using latent_size: {latent_size}")

    # Create generator with specified latent_size
    generator = Generator(
        num_channels=config.num_channels,
        latent_size=latent_size,
        feature_map_size=config.generator_feature_map_size
    ).to(device_info.backend)

    # Load trained weights
    try:
        state_dict = torch.load(
            generator_path, map_location=device_info.backend)
        generator.load_state_dict(state_dict)
        print(f"Loaded generator from {generator_path}")
    except Exception as e:
        print(f"Failed to load generator: {e}")
        print(
            f"Make sure the model was trained with latent_size={latent_size}")
        print("Or try a different --latent_size parameter")
        return

    generator.eval()

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Validate num_images
    if not (7000 <= num_images <= 10000):
        print(
            f"Warning: Generating {num_images} images (recommended: 7,000-10,000)")

    # Calculate batches needed
    num_batches = (num_images + batch_size - 1) // batch_size
    total_generated = 0

    print(f"Generating {num_images} Monet-style images...")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Generated size: 64x64x3 → Upscaled to: 256x256x3")
    print(f"Output format: 256x256x3 RGB JPG")

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Calculate how many images to generate in this batch
            remaining_images = num_images - total_generated
            current_batch_size = min(batch_size, remaining_images)

            if current_batch_size <= 0:
                break

            # Generate random noise
            z = torch.randn(current_batch_size, latent_size,
                            1, 1, device=device_info.backend)

            # Generate images
            fake_images = generator(z)

            # Process each image in the batch
            for i in range(current_batch_size):
                # Get single image tensor
                img_tensor = fake_images[i]  # Shape: (3, 64, 64)

                # Denormalize to [0, 255] uint8
                img_uint8 = denormalize_image(img_tensor)

                # Convert to numpy and transpose to (H, W, C)
                img_np = img_uint8.cpu().numpy().transpose(1, 2, 0)

                # Convert to PIL Image
                img_pil = Image.fromarray(img_np, mode='RGB')

                # Upscale from 64x64 to 256x256 using high-quality resampling
                img_pil_256 = img_pil.resize((256, 256), Image.LANCZOS)

                # Save as JPG with sequential numbering
                img_filename = f"{total_generated + i + 1:05d}.jpg"
                img_path = os.path.join(output_dir, img_filename)
                img_pil_256.save(img_path, 'JPEG', quality=95)

            total_generated += current_batch_size

    print(f"Successfully generated {total_generated} Monet-style images")
    print(f"Saved to: {output_dir}")
    print(f"Image format: 256x256x3 RGB JPG (upscaled from 64x64)")
    print(f"File naming: 00001.jpg to {total_generated:05d}.jpg")

    # Verify a few images
    sample_files = list(Path(output_dir).glob("*.jpg"))[:3]
    print("Sample verification:")
    for sample_file in sample_files:
        img = Image.open(sample_file)
        file_size = sample_file.stat().st_size / 1024  # KB
        print(f"{sample_file.name}: {img.size} {img.mode} ({file_size:.1f} KB)")

    print(
        f"Ready for submission! Total files: {len(list(Path(output_dir).glob('*.jpg')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Monet-style images for submission")
    parser.add_argument("--generator_path", type=str,
                        default="./artifacts/generator_final.pth",
                        help="Path to trained generator model (.pth file)")
    parser.add_argument("--output_dir", type=str,
                        default="./artifacts/submission_images",
                        help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=8000,
                        help="Number of images to generate (7000-10000)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--latent_size", type=int, default=128,
                        help="Size of latent vector (default: 128, same as main.py)")

    args = parser.parse_args()

    # Validate num_images range
    if not (7000 <= args.num_images <= 10000):
        print("Warning: num_images should be between 7,000 and 10,000 for submission")

    generate_submission_images(
        generator_path=args.generator_path,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        latent_size=args.latent_size
    )


if __name__ == "__main__":
    main()
