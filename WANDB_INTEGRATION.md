# Weights & Biases (wandb) Integration

This DCGAN implementation includes comprehensive Weights & Biases integration for experiment tracking, visualization, and monitoring.

## Features

### 🔧 Easy Setup
- **CLI Control**: Enable/disable wandb with `--use_wandb 1/0`
- **Environment Configuration**: Set project name via `WANDB_PROJECT_NAME` in `.env`
- **Backward Compatible**: Works perfectly with or without wandb enabled

### 📊 Training Metrics
The integration logs comprehensive training metrics:

- **Loss Tracking**: 
  - Discriminator loss (`train/discriminator_loss`)
  - Generator loss (`train/generator_loss`)
  - Epoch-level averages (`epoch_summary/discriminator_loss`, `epoch_summary/generator_loss`)

- **GAN-Specific Metrics**:
  - `train/Dx`: Discriminator output on real images
  - `train/DGz1`: Discriminator output on fake images during D training  
  - `train/DGz2`: Discriminator output on fake images during G training
  - `train/acc/D_real`: Discriminator accuracy on real images
  - `train/acc/D_fake`: Discriminator accuracy on fake images
  - `train/acc/D`: Overall discriminator accuracy

### 🖼️ Image Logging
- **Generated Samples**: Automatically logs generated images every epoch
- **Comparison Grids**: Side-by-side real vs fake image grids every 5 epochs
- **Progressive Visualization**: Track how your generator improves over time

### 🔍 Model Monitoring
- **Gradient Tracking**: Automatic gradient logging for both generator and discriminator
- **System Metrics**: GPU memory usage tracking
- **Hyperparameter Logging**: All training parameters automatically logged

### 📈 Advanced Evaluation
- **MiFID Integration**: Memorization-Informed Fréchet Inception Distance evaluation
- **Performance Optimized**: Disabled by default for faster training
- **Configurable Frequency**: Control evaluation frequency with `--mifid_eval_every_epochs`
- **Batch Control**: Limit evaluation batches with `--mifid_eval_batches`

## Usage

### Basic Usage

```bash
# Enable wandb logging
uv run python main.py gan --use_wandb 1

# Disable wandb logging  
uv run python main.py gan --use_wandb 0
```

### Advanced Configuration

```bash
# Full training with wandb (fast, no MiFID)
uv run python main.py gan \
    --use_wandb 1 \
    --epochs 100 \
    --batch_size 32 \
    --image_log_every_iters 50

# With MiFID evaluation (slower, but more comprehensive)
uv run python main.py gan \
    --use_wandb 1 \
    --epochs 100 \
    --batch_size 32 \
    --mifid_eval_every_epochs 20 \
    --mifid_eval_batches 3 \
    --image_log_every_iters 50
```

### Environment Setup

Create a `.env` file in the project root:

```bash
WANDB_PROJECT_NAME=dcgan-monet-style
WANDB_ENTITY=your-wandb-entity  # Optional
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_wandb` | 1 | Enable/disable wandb logging (0 or 1) |
| `--mifid_eval_every_epochs` | 0 | Evaluate MiFID every N epochs (0 disables, **disabled by default for performance**) |
| `--mifid_eval_batches` | 3 | Number of batches for MiFID evaluation |
| `--image_log_every_iters` | 10 | Log training metrics every N iterations |

## What Gets Logged

### Training Phase
- Real-time loss curves for both generator and discriminator
- GAN-specific metrics (D(x), D(G(z)), accuracies)
- System resource usage (GPU memory)
- Training progress and iteration counts

### Epoch Summaries  
- Average losses per epoch
- Generated sample images
- Model checkpoint information

### Evaluation Phase
- MiFID scores for quality assessment
- Comparative image grids (real vs fake)
- Progressive sample quality visualization

## Monitoring Your Training

### Key Metrics to Watch

1. **Loss Balance**: Generator and discriminator losses should be roughly balanced
2. **D(x) ≈ 0.5**: Discriminator output on real images should hover around 0.5
3. **D(G(z)) ≈ 0.5**: Discriminator output on fake images should approach 0.5
4. **MiFID Score**: Lower is better (measures image quality and diversity)

### Warning Signs
- **Mode Collapse**: Generator loss drops rapidly while discriminator loss increases
- **Training Instability**: Highly oscillating losses
- **Discriminator Dominance**: D(x) → 1, D(G(z)) → 0 consistently

## Implementation Details

### MonetWandb Class
The `MonetWandb` class in `monet_wandb.py` provides:
- Structured logging methods for different metric types
- Image preprocessing and grid creation
- MiFID evaluation with proper tensor handling
- Error handling and graceful degradation

### Integration Points
- **App Class**: Initializes wandb and passes logger to DCGAN
- **DCGAN Class**: Enhanced training loop with comprehensive metric logging
- **Training Methods**: Return detailed metrics for wandb logging

## Troubleshooting

### Common Issues

1. **"No module named 'wandb'"**
   ```bash
   uv sync  # Install dependencies
   ```

2. **Login Required**
   ```bash
   wandb login
   ```

3. **Project Not Found**
   - Check `WANDB_PROJECT_NAME` in `.env`
   - Ensure you have access to the wandb project

4. **Memory Issues During MiFID**
   - Reduce `--mifid_eval_batches`
   - Set `--mifid_eval_every_epochs 0` to disable

### Performance Tips
- **MiFID is disabled by default** for optimal training speed
- Use `--image_log_every_iters` to control logging frequency  
- Enable MiFID only when you need quality metrics: `--mifid_eval_every_epochs 20`
- Reduce MiFID batches for faster evaluation: `--mifid_eval_batches 2`
- Monitor GPU memory usage in wandb dashboard

## Example Wandb Dashboard

Your wandb dashboard will show:
- 📈 Real-time loss curves
- 🖼️ Generated image galleries  
- 📊 System resource usage
- 🎯 MiFID quality scores
- ⚙️ Complete hyperparameter tracking

This comprehensive integration helps you understand your GAN's training dynamics and achieve better results through data-driven insights.
