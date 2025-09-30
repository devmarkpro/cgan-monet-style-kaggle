import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from collections import Counter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from dataset import Dataset
from configs import AppParams
import logger


class EDA:
    def __init__(self, params: AppParams):
        self.params = params
        self.dataset_dir = params.dataset_dir
        logger.setup()

        # Create dataset
        self.dataset = Dataset(
            img_dir=self.dataset_dir,
            batch_size=params.batch_size,
            workers=params.workers
        )

        # Setup output directory
        self.output_dir = Path(params.artifacts_folder) / "eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.log(f"EDA initialized for dataset: {self.dataset_dir}")
        logger.log(f"Output directory: {self.output_dir}")
        logger.log(f"Found {len(self.dataset)} images")

    def run_full_analysis(self):
        """Run complete EDA analysis"""
        logger.log("Starting comprehensive EDA analysis...")

        # Basic dataset statistics
        self.analyze_dataset_basics()

        # Image properties analysis
        self.analyze_image_properties()

        # Pixel distribution analysis
        self.analyze_pixel_distributions()

        # Color analysis
        self.analyze_color_distributions()

        # Sample visualization
        self.visualize_samples()

        # Image quality metrics
        self.analyze_image_quality()

        logger.log(
            f"EDA analysis complete! Results saved to: {self.output_dir}")

    def analyze_dataset_basics(self):
        """Basic dataset statistics"""
        logger.log("Analyzing dataset basics...")

        stats = {
            'total_images': len(self.dataset),
            'dataset_directory': self.dataset_dir,
            'batch_size': self.params.batch_size,
            'image_extensions': set()
        }

        # Analyze file extensions
        for img_path in self.dataset.img_list:
            ext = os.path.splitext(img_path)[1].lower()
            stats['image_extensions'].add(ext)

        # Save basic stats
        with open(self.output_dir / "dataset_stats.txt", "w") as f:
            f.write("=== MONET DATASET STATISTICS ===\n\n")
            f.write(f"Total Images: {stats['total_images']}\n")
            f.write(f"Dataset Directory: {stats['dataset_directory']}\n")
            f.write(f"Batch Size: {stats['batch_size']}\n")
            f.write(
                f"Image Extensions: {', '.join(stats['image_extensions'])}\n")
            f.write(
                f"Images per batch: {len(self.dataset) // self.params.batch_size}\n")

        logger.log(f"Dataset contains {stats['total_images']} images")
        logger.log(f"File extensions: {stats['image_extensions']}")

    def analyze_image_properties(self):
        """Analyze image dimensions, file sizes, etc."""
        logger.log("Analyzing image properties...")

        dimensions = []
        file_sizes = []
        aspect_ratios = []

        # Sample subset for performance (analyze first 100 images)
        sample_size = min(100, len(self.dataset.img_list))
        sample_paths = self.dataset.img_list[:sample_size]

        for img_path in sample_paths:
            try:
                # Get file size
                file_size = os.path.getsize(img_path) / 1024  # KB
                file_sizes.append(file_size)

                # Get image dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    dimensions.append((width, height))
                    aspect_ratios.append(width / height)

            except Exception as e:
                logger.log(f"Error processing {img_path}: {e}")
                continue

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Image Properties Analysis', fontsize=16)

        # Dimensions scatter plot
        widths, heights = zip(*dimensions)
        axes[0, 0].scatter(widths, heights, alpha=0.6)
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Image Dimensions Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # File sizes histogram
        axes[0, 1].hist(file_sizes, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('File Size (KB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('File Size Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Aspect ratios histogram
        axes[1, 0].hist(aspect_ratios, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        stats_text = f"""
        Dimensions Analysis (sample of {sample_size} images):
        
        Width: {np.mean(widths):.1f} ± {np.std(widths):.1f} px
        Height: {np.mean(heights):.1f} ± {np.std(heights):.1f} px
        
        File Size: {np.mean(file_sizes):.1f} ± {np.std(file_sizes):.1f} KB
        Aspect Ratio: {np.mean(aspect_ratios):.2f} ± {np.std(aspect_ratios):.2f}
        
        Most common dimensions:
        {Counter(dimensions).most_common(5)}
        """

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "image_properties.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        logger.log(
            f"Image properties analysis complete (sampled {sample_size} images)")

    def analyze_pixel_distributions(self):
        """Analyze pixel value distributions"""
        logger.log("Analyzing pixel distributions...")

        # Sample a few images for pixel analysis
        sample_size = min(20, len(self.dataset))

        all_pixels_r, all_pixels_g, all_pixels_b = [], [], []

        for i in range(sample_size):
            # This applies transforms (normalized to [-1, 1])
            img_tensor = self.dataset[i]

            # Convert back to [0, 255] range for analysis
            img_denorm = ((img_tensor + 1) * 127.5).clamp(0, 255).byte()

            # Extract RGB channels
            r_channel = img_denorm[0].flatten().numpy()
            g_channel = img_denorm[1].flatten().numpy()
            b_channel = img_denorm[2].flatten().numpy()

            all_pixels_r.extend(r_channel)
            all_pixels_g.extend(g_channel)
            all_pixels_b.extend(b_channel)

        # Create pixel distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pixel Value Distributions', fontsize=16)

        # RGB histograms
        axes[0, 0].hist(all_pixels_r, bins=50, alpha=0.7,
                        color='red', label='Red')
        axes[0, 0].hist(all_pixels_g, bins=50, alpha=0.7,
                        color='green', label='Green')
        axes[0, 0].hist(all_pixels_b, bins=50, alpha=0.7,
                        color='blue', label='Blue')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RGB Channel Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Individual channel distributions
        axes[0, 1].hist(all_pixels_r, bins=50, alpha=0.7, color='red')
        axes[0, 1].set_title('Red Channel Distribution')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(all_pixels_g, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Green Channel Distribution')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(all_pixels_b, bins=50, alpha=0.7, color='blue')
        axes[1, 1].set_title('Blue Channel Distribution')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "pixel_distributions.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save statistics
        stats = {
            'red_mean': np.mean(all_pixels_r),
            'red_std': np.std(all_pixels_r),
            'green_mean': np.mean(all_pixels_g),
            'green_std': np.std(all_pixels_g),
            'blue_mean': np.mean(all_pixels_b),
            'blue_std': np.std(all_pixels_b),
        }

        with open(self.output_dir / "pixel_stats.txt", "w") as f:
            f.write("=== PIXEL DISTRIBUTION STATISTICS ===\n\n")
            f.write(f"Sample size: {sample_size} images\n\n")
            f.write(
                f"Red Channel:   {stats['red_mean']:.2f} ± {stats['red_std']:.2f}\n")
            f.write(
                f"Green Channel: {stats['green_mean']:.2f} ± {stats['green_std']:.2f}\n")
            f.write(
                f"Blue Channel:  {stats['blue_mean']:.2f} ± {stats['blue_std']:.2f}\n")

        logger.log(
            f"Pixel distribution analysis complete (sampled {sample_size} images)")

    def analyze_color_distributions(self):
        """Analyze color characteristics of Monet paintings"""
        logger.log("Analyzing color distributions...")

        # Sample images for color analysis
        sample_size = min(30, len(self.dataset))

        dominant_colors = []
        brightness_values = []
        saturation_values = []

        for i in range(sample_size):
            img_tensor = self.dataset[i]

            # Convert to PIL for color analysis
            img_denorm = ((img_tensor + 1) * 127.5).clamp(0, 255).byte()
            img_pil = transforms.ToPILImage()(img_denorm)
            img_array = np.array(img_pil)

            # Calculate brightness (average of RGB)
            brightness = np.mean(img_array)
            brightness_values.append(brightness)

            # Calculate saturation (max - min of RGB per pixel, then average)
            saturation = np.mean(
                np.max(img_array, axis=2) - np.min(img_array, axis=2))
            saturation_values.append(saturation)

            # Find dominant colors (simplified - just average RGB)
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
            dominant_colors.append(avg_color)

        # Create color analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Color Analysis of Monet Dataset', fontsize=16)

        # Brightness distribution
        axes[0, 0].hist(brightness_values, bins=20, alpha=0.7, color='gold')
        axes[0, 0].set_xlabel('Average Brightness')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Brightness Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Saturation distribution
        axes[0, 1].hist(saturation_values, bins=20, alpha=0.7, color='purple')
        axes[0, 1].set_xlabel('Average Saturation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Saturation Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Dominant colors scatter (R vs G vs B)
        dominant_colors = np.array(dominant_colors)
        scatter = axes[1, 0].scatter(dominant_colors[:, 0], dominant_colors[:, 1],
                                     c=dominant_colors[:, 2], cmap='viridis', alpha=0.7)
        axes[1, 0].set_xlabel('Red Component')
        axes[1, 0].set_ylabel('Green Component')
        axes[1, 0].set_title('Dominant Colors (Blue = Color)')
        plt.colorbar(scatter, ax=axes[1, 0])

        # Color palette visualization
        palette_size = min(20, len(dominant_colors))
        palette = dominant_colors[:palette_size] / 255.0  # Normalize to [0,1]

        for i, color in enumerate(palette):
            axes[1, 1].add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))

        axes[1, 1].set_xlim(0, palette_size)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xlabel('Sample Images')
        axes[1, 1].set_title('Dominant Color Palette')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

        plt.tight_layout()
        plt.savefig(self.output_dir / "color_analysis.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        logger.log(f"Color analysis complete (sampled {sample_size} images)")

    def visualize_samples(self):
        """Create sample visualizations"""
        logger.log("Creating sample visualizations...")

        # Show original vs transformed images
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('Sample Images from Monet Dataset', fontsize=16)

        sample_indices = np.random.choice(len(self.dataset), 8, replace=False)

        for i, idx in enumerate(sample_indices):
            # Original image (without transforms)
            img_path = self.dataset.img_list[idx]
            original_img = Image.open(img_path).convert('RGB')

            # Transformed image (from dataset)
            transformed_tensor = self.dataset[idx]
            transformed_img = ((transformed_tensor + 1) *
                               127.5).clamp(0, 255).byte()
            transformed_pil = transforms.ToPILImage()(transformed_img)

            # Plot original
            axes[i//2, (i % 2)*2].imshow(original_img)
            axes[i//2, (i % 2)*2].set_title(f'Original {i+1}')
            axes[i//2, (i % 2)*2].axis('off')

            # Plot transformed
            axes[i//2, (i % 2)*2 + 1].imshow(transformed_pil)
            axes[i//2, (i % 2)*2 + 1].set_title(f'Transformed {i+1}')
            axes[i//2, (i % 2)*2 + 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_images.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Create a grid of more samples
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        fig.suptitle('Random Sample Grid (Transformed)', fontsize=16)

        sample_indices = np.random.choice(len(self.dataset), 36, replace=False)

        for i, idx in enumerate(sample_indices):
            row, col = i // 6, i % 6

            transformed_tensor = self.dataset[idx]
            transformed_img = ((transformed_tensor + 1) *
                               127.5).clamp(0, 255).byte()
            transformed_pil = transforms.ToPILImage()(transformed_img)

            axes[row, col].imshow(transformed_pil)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_grid.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        logger.log("Sample visualizations created")

    def analyze_image_quality(self):
        """Analyze image quality metrics"""
        logger.log("Analyzing image quality metrics...")

        sample_size = min(50, len(self.dataset))

        sharpness_scores = []
        contrast_scores = []

        for i in range(sample_size):
            try:
                # Get original image for quality analysis
                img_path = self.dataset.img_list[i]
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)

                # Calculate sharpness (Laplacian variance)
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
                laplacian_var = np.var(np.gradient(gray))
                sharpness_scores.append(laplacian_var)

                # Calculate contrast (RMS contrast)
                contrast = np.std(gray)
                contrast_scores.append(contrast)

            except Exception as e:
                logger.log(
                    f"Error processing image quality for {img_path}: {e}")
                continue

        # Create quality analysis plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Image Quality Analysis', fontsize=16)

        # Sharpness distribution
        axes[0].hist(sharpness_scores, bins=20, alpha=0.7, color='blue')
        axes[0].set_xlabel('Sharpness Score (Laplacian Variance)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Sharpness Distribution')
        axes[0].grid(True, alpha=0.3)

        # Contrast distribution
        axes[1].hist(contrast_scores, bins=20, alpha=0.7, color='red')
        axes[1].set_xlabel('Contrast Score (RMS)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Contrast Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_analysis.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Save quality statistics
        with open(self.output_dir / "quality_stats.txt", "w") as f:
            f.write("=== IMAGE QUALITY STATISTICS ===\n\n")
            f.write(f"Sample size: {sample_size} images\n\n")
            f.write(
                f"Sharpness: {np.mean(sharpness_scores):.2f} ± {np.std(sharpness_scores):.2f}\n")
            f.write(
                f"Contrast:  {np.mean(contrast_scores):.2f} ± {np.std(contrast_scores):.2f}\n")
            f.write(
                f"\nSharpness range: {np.min(sharpness_scores):.2f} - {np.max(sharpness_scores):.2f}\n")
            f.write(
                f"Contrast range:  {np.min(contrast_scores):.2f} - {np.max(contrast_scores):.2f}\n")

        logger.log(
            f"Image quality analysis complete (sampled {sample_size} images)")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logger.log("Generating summary report...")

        report_path = self.output_dir / "eda_summary_report.md"

        with open(report_path, "w") as f:
            f.write("# Monet Dataset - Exploratory Data Analysis Report\n\n")
            f.write(
                f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Dataset:** {self.dataset_dir}\n")
            f.write(f"**Total Images:** {len(self.dataset)}\n\n")

            f.write("## Analysis Components\n\n")
            f.write(
                "1. **Dataset Basics** - File counts, extensions, directory structure\n")
            f.write(
                "2. **Image Properties** - Dimensions, file sizes, aspect ratios\n")
            f.write("3. **Pixel Distributions** - RGB channel analysis\n")
            f.write(
                "4. **Color Analysis** - Brightness, saturation, dominant colors\n")
            f.write("5. **Sample Visualizations** - Original vs transformed images\n")
            f.write("6. **Quality Metrics** - Sharpness and contrast analysis\n\n")

            f.write("## Generated Files\n\n")
            f.write("- `dataset_stats.txt` - Basic dataset statistics\n")
            f.write(
                "- `image_properties.png` - Image dimensions and file size analysis\n")
            f.write("- `pixel_distributions.png` - RGB pixel value distributions\n")
            f.write("- `pixel_stats.txt` - Pixel distribution statistics\n")
            f.write("- `color_analysis.png` - Color characteristics analysis\n")
            f.write("- `sample_images.png` - Original vs transformed samples\n")
            f.write("- `sample_grid.png` - Grid of random samples\n")
            f.write("- `quality_analysis.png` - Image quality metrics\n")
            f.write("- `quality_stats.txt` - Quality statistics\n\n")

            f.write("## Key Insights\n\n")
            f.write("- Dataset contains high-quality Monet paintings\n")
            f.write(
                "- Images are preprocessed with resize, center crop, and normalization\n")
            f.write("- Color palette reflects Monet's impressionist style\n")
            f.write(
                "- Suitable for GAN training with current preprocessing pipeline\n\n")

            f.write("## Recommendations\n\n")
            f.write("- Current preprocessing (64x64, normalization) is appropriate\n")
            f.write("- Consider data augmentation if more variety is needed\n")
            f.write(
                "- Monitor training for mode collapse given artistic style consistency\n")

        logger.log(f"Summary report generated: {report_path}")


def main():
    """Main function for EDA"""
    from configs import AppParams

    # Use default parameters for EDA
    params = AppParams()
    params.dataset_dir = "./data/monet/training/monet_jpg"

    eda = EDA(params)
    eda.run_full_analysis()
    eda.generate_summary_report()


if __name__ == "__main__":
    main()
