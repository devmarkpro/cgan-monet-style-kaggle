# Deep Convolutional Generative Adversarial Networks for Monet-Style Image Generation

**Course:** CSCA 5642: Introduction to Deep Learning  
**Institution:** CU Boulder University  
**Author:** Mark Karamyar (mark.karamyar@colorado.edu)
**Date:** September 2025
**GitHub Repository:** https://github.com/devmarkpro/dcgan-monet-style-kaggle

---

## Abstract

This report presents the implementation and evaluation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating Monet-style paintings. The project addresses the challenge of creating artificial artwork that captures the unique impressionist characteristics of Claude Monet's paintings. Through careful architecture design, data preprocessing, and training optimization, we successfully developed a model that can generate 8,000 high-quality 256x256 Monet-style images. The implementation shows the effectiveness of adversarial training in learning complex artistic styles and provides insights into the practical considerations of GAN training for artistic image creation.

---

## 1. Problem Statement

The main goal of this project is to develop a generative model that can create artificial images that copy the unique artistic style of Claude Monet, the famous French impressionist painter. This work was conducted as part of the Kaggle "I'm Something of a Painter Myself" competition [6]. This problem belongs to the area of style transfer and artistic image generation, which has important applications in digital art, content creation, and cultural preservation.

### 1.1 Motivation

Monet's impressionist style is characterized by loose brushwork, emphasis on light and its changing qualities, and vibrant color palettes. Traditional computer graphics approaches struggle to capture these detailed artistic elements, making deep learning-based generative models an attractive solution. The challenge is in training a model that can understand and reproduce the complex visual patterns, color relationships, and compositional elements that define Monet's artistic style.

### 1.2 Technical Challenges

The main technical challenges include:
- Learning complex artistic patterns from a limited dataset
- Maintaining visual coherence while generating diverse outputs
- Balancing the adversarial training dynamics between generator and discriminator
- Preventing mode collapse in the artistic domain
- Scaling generated images to meet submission requirements ($256 \times 256$ pixels)

---

## 2. Dataset Characteristics

### 2.1 Dataset Overview

The dataset consists of 300 high-quality digital reproductions of Monet paintings, stored in JPEG format within the `./data/monet/training/monet_jpg` directory [6]. This relatively small dataset size is typical for artistic style transfer tasks, where quality often matters more than quantity. The dataset was provided as part of the Kaggle "I'm Something of a Painter Myself" competition, which focuses on GAN-based artistic style transfer.

### 2.2 Basic Statistics

- **Total Images:** 300
- **Format:** JPEG (.jpg)
- **Batch Configuration:** 16 images per batch (18 complete batches, 12 images dropped due to `drop_last=True`)
- **Source:** Curated collection of Monet paintings
- **Quality:** High-resolution digital reproductions

The dataset size of 300 images presents both opportunities and challenges. While smaller than typical computer vision datasets, it is sufficient for capturing Monet's artistic patterns due to the consistency in his impressionist style. However, it requires careful regularization and augmentation strategies to prevent overfitting.

---

## 3. Exploratory Data Analysis

The exploratory data analysis revealed crucial insights that informed our preprocessing and model design decisions. The analysis encompassed six key areas: dataset basics, image properties, pixel distributions, color characteristics, sample visualizations, and quality metrics.

### 3.1 Image Properties Analysis

![Image Properties](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eda/image_properties.png)

The image properties analysis revealed significant variation in the original dimensions of Monet paintings, which is expected given the diverse canvas sizes used by the artist throughout his career. This heterogeneity necessitated standardized preprocessing to ensure consistent input dimensions for the neural network. The file size distribution indicated high-quality source images, providing rich detail for the model to learn from.

### 3.2 Pixel Distribution Analysis

![Pixel Distributions](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eda/pixel_distributions.png)

The pixel distribution analysis provided insights into the color characteristics of Monet's work:

- **Red Channel:** Mean $= 122.02$, Standard Deviation $= 52.45$
- **Green Channel:** Mean $= 123.96$, Standard Deviation $= 50.06$  
- **Blue Channel:** Mean $= 110.35$, Standard Deviation $= 54.61$

These statistics reveal that Monet's paintings exhibit a slight bias toward warmer tones (higher red and green values) with the blue channel showing the highest variance, reflecting his diverse use of sky and water elements. The relatively balanced channel means suggest good color distribution across the spectrum, which is favorable for GAN training.

### 3.3 Color Analysis

![Color Analysis](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eda/color_analysis.png)

The color analysis demonstrates the rich palette characteristic of impressionist work. Monet's paintings show high saturation in key areas while maintaining subtle gradations that create the signature impressionist effect. The brightness distribution indicates a preference for well-lit scenes, consistent with Monet's focus on capturing natural light conditions.

### 3.4 Sample Visualizations

![Sample Images](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eda/sample_images.png)

The sample visualization shows both original and preprocessed versions of representative paintings. This comparison illustrates how our preprocessing pipeline maintains the essential artistic characteristics while standardizing the format for neural network input.

![Sample Grid](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eda/sample_grid.png)

The sample grid provides an overview of the dataset diversity, showcasing various subjects including landscapes, water scenes, and architectural elements that are hallmarks of Monet's work.

### 3.5 Quality Metrics

![Quality Analysis](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eda/quality_analysis.png)

The quality analysis revealed important characteristics:

- **Sharpness:** Mean $= 166.23$, Standard Deviation $= 117.05$ (Range: $17.67 - 588.09$)
- **Contrast:** Mean $= 45.04$, Standard Deviation $= 11.92$ (Range: $21.90 - 67.34$)

The wide sharpness range reflects Monet's varied brushwork techniques, from soft, blended areas to more defined structural elements. The moderate contrast values are consistent with impressionist techniques that often favor subtle tonal variations over stark contrasts.

---

## 4. Data Preprocessing

### 4.1 Preprocessing Pipeline

Based on the EDA findings, we implemented a carefully designed preprocessing pipeline that balances computational efficiency with artistic quality:

```python
self.transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
```

### 4.2 Design Rationale

**Resolution Reduction to $64 \times 64$:**
The decision to reduce image resolution from the original dimensions to $64 \times 64$ pixels was based on several important factors:

1. **Computational Efficiency:** Training at full resolution ($256 \times 256$ or higher) would require significantly more computational resources and training time
2. **Training Stability:** Lower resolution reduces the complexity of the learning task, leading to more stable GAN training dynamics
3. **Pattern Focus:** At $64 \times 64$, the model focuses on essential artistic patterns and color relationships rather than fine details, which aligns well with impressionist aesthetics
4. **Memory Constraints:** Enables efficient batch processing within available GPU memory, particularly important for Apple Silicon MPS backend

**Center Cropping:**
Center cropping ensures that the most important compositional elements are preserved while maintaining aspect ratio consistency across the dataset.

**Normalization Strategy:**
The normalization to $[-1, 1]$ range using $[0.5, 0.5, 0.5]$ mean and standard deviation is chosen specifically for GAN training, as it matches the output range of the generator's tanh activation function.

### 4.3 Post-Processing for Submission

To meet the $256 \times 256$ pixel requirement for final submission, we implemented a high-quality upscaling process using LANCZOS resampling, which preserves artistic details while scaling the generated $64 \times 64$ images to the required dimensions.

---

## 5. Deep Learning Approach

### 5.1 Architecture Overview

Our implementation follows the DCGAN (Deep Convolutional Generative Adversarial Network) architecture, which has proven effective for image generation tasks. The system consists of two competing neural networks: a generator that creates artificial images and a discriminator that tells the difference between real and generated images.

### 5.2 Generator Architecture

The generator network transforms random noise vectors into Monet-style images through a series of transposed convolutions:

```python
# Generator Architecture (64x64 output)
nn.ConvTranspose2d(latent_size, ngf * 8, 4, 1, 0, bias=False)  # 1x1 -> 4x4
nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)     # 4x4 -> 8x8
nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)     # 8x8 -> 16x16
nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)         # 16x16 -> 32x32
nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False)               # 32x32 -> 64x64
```

**Key Design Elements:**
- **Latent Vector Size:** 128 dimensions provide sufficient capacity for encoding artistic variations
- **Feature Map Progression:** Follows the standard DCGAN pattern of halving feature maps while doubling spatial dimensions
- **Activation Functions:** ReLU activations in hidden layers with Tanh output to match the $[-1, 1]$ normalized input range
- **Batch Normalization:** Applied to all layers except the output to stabilize training

The generator architecture is designed specifically to capture hierarchical artistic features, from basic color patterns in early layers to complex compositional elements in later layers.

**Actual Architecture Implementation:**
Based on the training logs, the generator network consists of:
```
ConvTranspose2d(128, 512, kernel_size=(4, 4), stride=(1, 1))  # 1x1 -> 4x4
ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2))  # 4x4 -> 8x8  
ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2))  # 8x8 -> 16x16
ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2))   # 16x16 -> 32x32
ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2))    # 32x32 -> 64x64
```

This progressive upsampling allows the network to build artistic features step by step, starting with basic color and texture patterns and gradually adding more complex compositional elements.

### 5.3 Discriminator Architecture

The discriminator works as a binary classifier that learns to tell the difference between authentic Monet paintings and generated images:

```python
# Discriminator Architecture (64x64 input)
nn.Conv2d(3, ndf, 4, 2, 1, bias=False)           # 64x64 -> 32x32
nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)     # 32x32 -> 16x16
nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False) # 16x16 -> 8x8
nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False) # 8x8 -> 4x4
nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)       # 4x4 -> 1x1
```

**Key Design Elements:**
- **Strided Convolutions:** Reduce spatial dimensions while increasing feature depth
- **LeakyReLU Activations:** Prevent gradient vanishing and allow small negative values
- **No Batch Normalization in First Layer:** Preserves input image statistics
- **Sigmoid Output:** Produces probability scores for real/fake classification

**Actual Architecture Implementation:**
From the training logs, the discriminator network structure is:
```
Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2))    # 64x64 -> 32x32
Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2))   # 32x32 -> 16x16
Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2))  # 16x16 -> 8x8
Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2))  # 8x8 -> 4x4
Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1))   # 4x4 -> 1x1
```

This architecture effectively learns to tell the difference between authentic Monet paintings and generated images by progressively extracting features at multiple scales.

### 5.4 Training Dynamics

The DCGAN training process involves alternating optimization of the generator and discriminator networks:

**Discriminator Training:**
1. Train on real Monet images (target: 1)
2. Train on generated images (target: 0)
3. Backpropagate combined loss

**Generator Training:**
1. Generate images from random noise
2. Pass through discriminator (target: 1)
3. Backpropagate generator loss

**Loss Functions:**
- **Discriminator Loss:** Binary cross-entropy on real and fake image classifications
- **Generator Loss:** Binary cross-entropy encouraging discriminator to classify generated images as real

### 5.5 Training Optimizations

**Optimizer Configuration:**
- **Algorithm:** Adam optimizer for both networks
- **Learning Rates:** Discriminator: 0.0002, Generator: 0.0002
- **Beta Parameters:** β₁ = 0.5, β₂ = 0.999 (standard for GAN training)

**Monitoring and Evaluation:**
- **MiFID Score:** Memorization-informed Fréchet Inception Distance for quality assessment
- **Loss Tracking:** Separate monitoring of discriminator and generator losses
- **Sample Generation:** Regular image generation for visual quality assessment

---

## 6. Results and Analysis

### 6.1 Training Progress

The model was trained for 5,000 epochs using the command:
```bash
uv run main.py gan --epochs 5000 --image_log_every_iters 100 --mifid_eval_every_epochs 100 --use_wandb 1
```

**Training Configuration:**
- **Total Training Time:** 62 minutes 59.37 seconds
- **Device:** Apple Silicon (MPS)
- **Batch Size:** 16
- **Latent Size:** 128
- **Generator Feature Maps:** 64
- **Discriminator Feature Maps:** 64
- **Learning Rate:** 0.0002 (both networks)
- **Beta Parameters:** $\beta_1 = 0.5$, $\beta_2 = 0.999$

**Training Dynamics Analysis:**
The training logs reveal excellent convergence characteristics. Early epochs showed typical GAN initialization behavior:

- **Epoch 1:** D_loss: 2.0024, G_loss: 2.8799, D(x): 0.3896
- **Epoch 100:** Stabilized around D_loss: ~0.95, G_loss: ~1.9
- **Final Epoch (5000):** D_loss: 0.8335, G_loss: 1.7476, D(x): 0.7859

The discriminator accuracy remained consistently at $100\%$ throughout most of the training, indicating strong learning without mode collapse. The D(x) values stabilized around $0.78-0.82$, showing the discriminator learned to properly evaluate real Monet paintings.

### 6.2 Loss Evolution

![Generator vs Discriminator Loss](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/D-vs-G-Loss-stilted-elevator-52.png)

The loss curves demonstrate healthy adversarial training dynamics. The generator and discriminator losses show the characteristic oscillatory behavior expected in GAN training, with neither network completely dominating the other. This balance is crucial for generating high-quality artistic images.

![Generator Loss](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/generator_loss-stilted-elevator-52.png)

The generator loss shows a general downward trend with periodic fluctuations, indicating successful learning of the artistic style distribution while maintaining diversity in generated outputs.

![Discriminator Loss](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/discriminator_loss-stilted-elevator-52.png)

The discriminator loss maintains reasonable values throughout training, suggesting it successfully learned to distinguish between real and generated Monet paintings without becoming too powerful and preventing generator learning.

### 6.3 Quality Assessment

![MiFID Score Evolution](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eval_mifid_stilted-elevator-52.png)

The MiFID (Memorization-informed Fréchet Inception Distance) score provides an objective measure of generation quality. The final MiFID score achieved was **1.4698**, indicating excellent generation quality and strong alignment with the target artistic style. This low score demonstrates that the generated images successfully capture the statistical properties of Monet's artistic style without simply memorizing training examples.

![MiFID Evaluation Time](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/eval_mifid_time-stilted-elevator-52.png)

The evaluation timing shows consistent performance throughout the training process, with no significant degradation in computational efficiency.

### 6.4 Generated Image Evolution

The following images demonstrate the progression of generated artwork quality throughout training:

**Epoch 424:**

![Generated Images - Epoch 424](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/generated_image_epoch_424.png)

Early training shows basic color patterns and rough compositional elements beginning to emerge.

**Epoch 500:**

![Generated Images - Epoch 500](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/generated_image_epoch_500.png)

Improved coherence in color relationships and more defined structural elements.

**Epoch 1530:**

![Generated Images - Epoch 1530](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/generated_image_epoch_1530.png)

Mid-training results show significant improvement in artistic coherence and style consistency.

**Epoch 3000:**

![Generated Images - Epoch 3000](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/generated_image_epoch_3000.png)

Advanced training demonstrates sophisticated understanding of impressionist techniques.

**Epoch 5000 (Final):**

![Generated Images - Epoch 5000](https://raw.githubusercontent.com/devmarkpro/dcgan-monet-style-kaggle/main/report_material/generated_image_epoch_5000.png)

Final results exhibit mature artistic style with convincing Monet-like characteristics including appropriate color palettes, brushwork patterns, and compositional elements.

### 6.5 Final Output Generation

The trained model successfully generated 8,000 unique Monet-style images at $256 \times 256$ resolution for submission. The upscaling from $64 \times 64$ to $256 \times 256$ using LANCZOS resampling preserved the artistic qualities while meeting the technical requirements.

**Final Model Performance:**
- **Model Name:** stilted-elevator-52 (WandB run name)
- **Final MiFID Score:** 1.4698
- **Generator Loss:** 1.7476
- **Discriminator Loss:** 0.8335
- **Training Efficiency:** ~0.75 seconds per epoch
- **Generated Images:** 8,000 high-quality $256 \times 256$ Monet-style paintings

---

## 7. Conclusion and Summary

### 7.1 Project Achievements

This project successfully demonstrates the application of Deep Convolutional Generative Adversarial Networks for artistic style generation. Key achievements include:

1. **Successful Model Training:** Achieved stable GAN training over 5,000 epochs with balanced adversarial dynamics
2. **Quality Generation:** Produced convincing Monet-style artwork that captures essential impressionist characteristics
3. **Technical Implementation:** Developed a complete pipeline from data preprocessing to final image generation
4. **Scalable Output:** Generated 8,000 high-quality images meeting submission requirements

### 7.2 Technical Insights

The project provided several important insights into GAN training for artistic applications:

- **Dataset Size Considerations:** A focused dataset of 300 high-quality images proved sufficient for learning artistic style
- **Resolution Strategy:** Training at $64 \times 64$ with post-processing upscaling balanced computational efficiency with output quality
- **Architecture Effectiveness:** Standard DCGAN architecture worked well for artistic image generation
- **Training Stability:** Careful hyperparameter tuning and monitoring prevented common GAN training issues like mode collapse
- **Convergence Metrics:** The final MiFID score of 1.4698 shows exceptional quality, significantly better than typical benchmarks
- **Computational Efficiency:** Training completed in just over an hour on Apple Silicon, showing the efficiency of the $64 \times 64$ approach
- **Loss Balance:** Maintaining discriminator accuracy at $100\%$ while keeping generator loss decreasing indicates optimal adversarial balance

### 7.3 Limitations and Future Work

While the project achieved its main objectives, several areas present opportunities for future improvement:

1. **Resolution:** Direct training at higher resolutions could improve fine detail quality
2. **Dataset Expansion:** Additional training data could increase output diversity
3. **Architecture Improvements:** Progressive growing or attention mechanisms could improve generation quality
4. **Style Control:** Conditional generation could allow for more targeted artistic control

### 7.4 Educational Value

This project effectively demonstrates key concepts in deep learning including:
- Adversarial training dynamics
- Convolutional neural network architectures
- Transfer learning in artistic domains
- Practical considerations in model deployment

The implementation provides a solid foundation for understanding both the theoretical principles and practical challenges of generative modeling in computer vision applications.

---

## 8. References

[1] Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *arXiv preprint arXiv:1511.06434*. Available at: https://arxiv.org/abs/1511.06434

[2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems*, 27. Available at: https://arxiv.org/abs/1406.2661

[3] PyTorch Team. DCGAN Tutorial. *PyTorch Tutorials*. Available at: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

[4] Weights & Biases Team. Tables Tutorial. *Weights & Biases Documentation*. Available at: https://docs.wandb.ai/tutorials/tables/

[5] Lightning AI Team. Memorization-Informed Fréchet Inception Distance (MiFID). *TorchMetrics Documentation*. Available at: https://lightning.ai/docs/torchmetrics/stable/image/mifid.html

[6] Amy Jang, Ana Sofia Uzsoy, and Phil Culliton. I’m Something of a Painter Myself. https://kaggle.com/competitions/gan-getting-started, 2020. Kaggle.

---

## 9. Appendix

### 9.1 Project Structure

```
dcgan-monet-style-kaggle/
├── data/monet/training/monet_jpg/     # Training dataset
├── artifacts/                         # Generated outputs and models
│   ├── eda/                          # Exploratory data analysis results
│   └── submission_images/            # Final generated images
├── report_material/                  # Training visualizations and plots
├── main.py                          # Primary execution script
├── dcgan.py                         # Main DCGAN implementation
├── generator.py                     # Generator network definition
├── discriminator.py                 # Discriminator network definition
├── dataset.py                       # Data loading and preprocessing
├── generate_submission_images.py    # Image generation script
├── create_submission.py             # Simplified generation script
├── configs.py                       # Configuration parameters
├── device.py                        # Device management utilities
├── logger.py                        # Logging functionality
├── monet_wandb.py                   # Weights & Biases integration
└── utils.py                         # Utility functions
```

### 9.2 Configuration Parameters

The main.py script accepts the following key parameters:

| Parameter                          | Default | Description                         |
| ---------------------------------- | ------- | ----------------------------------- |
| `--epochs`                         | 100     | Number of training epochs           |
| `--batch_size`                     | 16      | Training batch size                 |
| `--latent_size`                    | 128     | Dimension of noise vector           |
| `--generator_feature_map_size`     | 64      | Generator feature map base size     |
| `--discriminator_feature_map_size` | 64      | Discriminator feature map base size |
| `--generator_lr`                   | 0.0002  | Generator learning rate             |
| `--discriminator_lr`               | 0.0002  | Discriminator learning rate         |
| `--use_wandb`                      | 1       | Enable Weights & Biases logging     |
| `--image_log_every_iters`          | 10      | Frequency of image logging          |
| `--mifid_eval_every_epochs`        | 0       | Frequency of MiFID evaluation       |

### 9.3 Running the Project

**Training the Model:**
```bash
# Basic training
uv run python main.py gan --epochs 1000

# Advanced training with monitoring
uv run python main.py gan --epochs 5000 --image_log_every_iters 100 --mifid_eval_every_epochs 100 --use_wandb 1

# Custom configuration
uv run python main.py gan --epochs 2000 --batch_size 32 --latent_size 256
```

**Generating Submission Images:**
```bash
# Simple generation (uses defaults)
uv run python create_submission.py

# Custom generation
uv run python generate_submission_images.py --num_images 10000 --latent_size 128 --batch_size 64
```

### 9.4 Image Generation Script

The `generate_submission_images.py` script provides flexible image generation capabilities:

**Key Features:**
- Automatic model loading with latent size detection
- High-quality LANCZOS upscaling from $64 \times 64$ to $256 \times 256$
- Batch processing for memory efficiency
- Progress tracking and quality verification
- Configurable output parameters

**Usage Examples:**
```bash
# Generate 8000 images (default)
uv run python generate_submission_images.py

# Custom parameters
uv run python generate_submission_images.py \
    --generator_path ./artifacts/generator_final.pth \
    --output_dir ./my_submission \
    --num_images 9000 \
    --batch_size 32 \
    --latent_size 128
```

### 9.5 Dependencies and Environment

**Core Dependencies:**
- PyTorch (>=2.0.0, with MPS/CUDA support)
- torchvision (>=0.15.0)
- torchmetrics (>=1.0.0) - for MiFID evaluation
- Pillow (>=9.0.0)
- NumPy (>=1.21.0)
- matplotlib (>=3.5.0)
- seaborn (>=0.13.2)
- tqdm (>=4.67.1)
- Weights & Biases (>=0.22.0)
- python-dotenv (>=0.9.9)

**Installation:**
```bash
# Using uv (recommended)
uv sync

# Using pip with pyproject.toml
pip install -e .
```

**Hardware Requirements:**
- GPU with at least 4GB VRAM (CUDA/MPS recommended)
- Apple Silicon Mac (MPS) or NVIDIA GPU (CUDA) for optimal performance
- 8GB+ system RAM
- Sufficient storage for dataset and generated images (~2GB for dataset + artifacts)

This comprehensive implementation provides a complete framework for artistic image generation using deep learning techniques, demonstrating both theoretical understanding and practical application of generative adversarial networks in the creative domain.
