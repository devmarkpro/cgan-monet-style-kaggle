#!/usr/bin/env python3
"""
Simple script to create submission images after training.
Just run: uv run python create_submission.py
"""

import os
from generate_submission_images import generate_submission_images


def main():
    print("Creating Monet-style images for submission...")
    run_name = "stilted-elevator-52"
    generator_path = f"./artifacts/generator_{run_name}.pth"
    output_dir = "./artifacts/submission_images"
    num_images = 8000
    batch_size = 64
    latent_size = 128

    if not os.path.exists(generator_path):
        print(f"Generator model not found at {generator_path}")
        print("Please train the model first:")
        print("uv run python main.py gan --epochs 500 --batch_size 128")
        return

    generate_submission_images(
        generator_path=generator_path,
        output_dir=output_dir,
        num_images=num_images,
        batch_size=batch_size,
        latent_size=latent_size
    )

    print("Submission ready!")
    print(f"Images location: {output_dir}")
    print(f"Total images: {num_images}")
    print(f"Format: JPG files named 00001.jpg to {num_images:05d}.jpg")


if __name__ == "__main__":
    main()
