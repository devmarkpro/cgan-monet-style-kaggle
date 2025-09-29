import wandb
import torch
import torchvision.utils as vutils
from dataclasses import asdict
from typing import Optional, Dict, Any
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance as MiFID

from configs import AppParams


class MonetWandb:
    def __init__(self, project_name: str, params: AppParams):
        self.project_name = project_name
        self.params = params
        self._setup()

    def _setup(self):
        self.run = wandb.init(
            # Set the wandb project where this run will be logged.
            project=self.project_name,
            # Track hyperparameters and run metadata.
            config=asdict(self.params),
        )
    
    def log_training_losses(self, d_loss: float, g_loss: float, epoch: int, iteration: int, 
                           d_x: Optional[float] = None, d_gz1: Optional[float] = None, 
                           d_gz2: Optional[float] = None, d_real_acc: Optional[float] = None,
                           d_fake_acc: Optional[float] = None, step: Optional[int] = None):
        """
        Log discriminator and generator losses during training with additional GAN metrics.
        
        Args:
            d_loss: Discriminator loss value
            g_loss: Generator loss value  
            epoch: Current epoch number
            iteration: Current iteration within epoch
            d_x: Discriminator output on real images (optional)
            d_gz1: Discriminator output on fake images during D training (optional)
            d_gz2: Discriminator output on fake images during G training (optional)
            d_real_acc: Discriminator accuracy on real images (optional)
            d_fake_acc: Discriminator accuracy on fake images (optional)
            step: Global step counter (optional)
        """
        if step is None:
            step = epoch * 1000 + iteration  # Simple step calculation
            
        log_data = {
            "train/discriminator_loss": d_loss,
            "train/generator_loss": g_loss,
            "train/epoch": epoch,
            "train/iteration": iteration,
        }
        
        # Add optional GAN-specific metrics
        if d_x is not None:
            log_data["train/Dx"] = d_x
        if d_gz1 is not None:
            log_data["train/DGz1"] = d_gz1
        if d_gz2 is not None:
            log_data["train/DGz2"] = d_gz2
        if d_real_acc is not None:
            log_data["train/acc/D_real"] = d_real_acc
        if d_fake_acc is not None:
            log_data["train/acc/D_fake"] = d_fake_acc
        if d_real_acc is not None and d_fake_acc is not None:
            log_data["train/acc/D"] = 0.5 * (d_real_acc + d_fake_acc)
        
        self.run.log(log_data, step=step)
    
    def log_epoch_summary(self, epoch: int, avg_d_loss: float, avg_g_loss: float, 
                         samples: Optional[torch.Tensor] = None):
        """
        Log epoch-level summary metrics and optionally generated samples.
        
        Args:
            epoch: Current epoch number
            avg_d_loss: Average discriminator loss for the epoch
            avg_g_loss: Average generator loss for the epoch
            samples: Generated samples tensor (optional)
        """
        log_data = {
            "epoch_summary/discriminator_loss": avg_d_loss,
            "epoch_summary/generator_loss": avg_g_loss,
            "epoch_summary/epoch": epoch,
        }
        
        # Log sample images if provided
        if samples is not None:
            try:
                # Create a grid of sample images
                img_grid = vutils.make_grid(samples.cpu(), padding=2, normalize=True, nrow=4)
                log_data["samples/generated_images"] = wandb.Image(
                    img_grid, caption=f"Generated samples - Epoch {epoch}"
                )
            except Exception as e:
                print(f"Failed to log sample images: {e}")
        
        self.run.log(log_data, step=epoch)
    
    def log_model_gradients(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        """
        Set up gradient logging for the models.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
        """
        try:
            wandb.watch(generator, log="gradients", log_freq=100)
            wandb.watch(discriminator, log="gradients", log_freq=100)
        except Exception as e:
            print(f"Failed to set up gradient logging: {e}")
    
    def log_system_metrics(self, gpu_memory_allocated: Optional[float] = None, 
                          gpu_memory_reserved: Optional[float] = None):
        """
        Log system metrics like GPU memory usage.
        
        Args:
            gpu_memory_allocated: GPU memory allocated in MB
            gpu_memory_reserved: GPU memory reserved in MB
        """
        log_data = {}
        
        if gpu_memory_allocated is not None:
            log_data["system/gpu_memory_allocated_mb"] = gpu_memory_allocated
            
        if gpu_memory_reserved is not None:
            log_data["system/gpu_memory_reserved_mb"] = gpu_memory_reserved
            
        if log_data:
            self.run.log(log_data)
    
    def log_custom_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log custom metrics dictionary.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step counter
        """
        self.run.log(metrics, step=step)
    
    def log_image_grids(self, fake_samples: torch.Tensor, real_samples: Optional[torch.Tensor] = None, 
                       step: Optional[int] = None, caption: str = ""):
        """
        Log both fake and real image grids for comparison.
        
        Args:
            fake_samples: Generated fake images tensor
            real_samples: Real images tensor (optional)
            step: Optional step counter
            caption: Caption for the images
        """
        log_data = {}
        
        try:
            # Log fake samples
            fake_grid = vutils.make_grid(fake_samples.cpu(), padding=2, normalize=True, nrow=8)
            log_data["samples/fake_grid"] = wandb.Image(fake_grid, caption=f"Fake {caption}")
            
            # Log real samples for comparison if provided
            if real_samples is not None:
                real_grid = vutils.make_grid(
                    real_samples[:min(64, real_samples.size(0))].cpu(), 
                    padding=2, normalize=True, nrow=8
                )
                log_data["samples/real_grid"] = wandb.Image(real_grid, caption=f"Real {caption}")
            
            self.run.log(log_data, step=step)
            
        except Exception as e:
            print(f"Failed to log image grids: {e}")
    
    def evaluate_mifid(self, generator, dataloader, latent_size: int, device, 
                      num_batches: Optional[int] = None, step: Optional[int] = None):
        """
        Evaluate MiFID (Memorization-Informed Fréchet Inception Distance) score.
        
        Args:
            generator: Generator model
            dataloader: DataLoader for real images
            latent_size: Size of the latent vector
            device: Device to run evaluation on
            num_batches: Number of batches to evaluate (optional, uses all if None)
            step: Optional step counter for logging
        """
        try:
            # Initialize MiFID metric
            mifid = MiFID(feature=2048, normalize=False, reset_real_features=False)
            
            # Determine number of batches to evaluate
            if num_batches is None:
                num_batches = len(dataloader)
            else:
                num_batches = min(num_batches, len(dataloader))
            
            generator.eval()
            with torch.no_grad():
                for i, (real_batch, _) in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    
                    # Process real images
                    real_batch = real_batch.to(device)
                    real_uint8 = self._denorm_to_uint8(real_batch).cpu()
                    mifid.update(real_uint8, real=True)
                    
                    # Generate fake images
                    batch_size = real_batch.size(0)
                    z = torch.randn(batch_size, latent_size, 1, 1, device=device)
                    fake_batch = generator(z).detach()
                    fake_uint8 = self._denorm_to_uint8(fake_batch).cpu()
                    mifid.update(fake_uint8, real=False)
            
            # Compute MiFID score
            mifid_score = float(mifid.compute().item())
            
            # Log to wandb
            log_data = {
                "eval/mifid": mifid_score,
                "eval/num_batches": num_batches,
            }
            
            self.run.log(log_data, step=step)
            
            return mifid_score
            
        except Exception as e:
            print(f"MiFID evaluation failed: {e}")
            return None
    
    def _denorm_to_uint8(self, tensor):
        """
        Convert normalized tensor [-1, 1] to uint8 [0, 255].
        
        Args:
            tensor: Normalized tensor with values in [-1, 1]
            
        Returns:
            Tensor with values in [0, 255] as uint8
        """
        # Denormalize from [-1, 1] to [0, 1]
        denorm = (tensor + 1.0) / 2.0
        # Clamp to [0, 1] and convert to [0, 255]
        denorm = torch.clamp(denorm, 0.0, 1.0)
        # Convert to uint8
        uint8_tensor = (denorm * 255).to(torch.uint8)
        return uint8_tensor
    
    def finish(self):
        """
        Finish the wandb run.
        """
        self.run.finish()
