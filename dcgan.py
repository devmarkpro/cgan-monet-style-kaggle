from typing import Any, Optional

import torch
from torch import nn, optim
from torch.optim import Adam

import device
import logger
import utils
from configs import AppParams
from dataset import Dataset
import numpy as np


class DCGAN:
    def __init__(self, generator, discriminator, dataset: Dataset, configs: AppParams, wandb_logger=None):
        self.config = configs
        self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator
        self.wandb_logger = wandb_logger

        self.device_info = device.get_device_info()

        self.generator.to(self.device_info.backend)
        self.discriminator.to(self.device_info.backend)

        self.discriminator.apply(utils.weights_init)
        self.generator.apply(utils.weights_init)

        # Set up gradient logging if wandb is available
        if self.wandb_logger:
            self.wandb_logger.log_model_gradients(
                self.generator, self.discriminator)

    def __str__(self):
        return f"{self.generator} \n\n {self.discriminator}"

    def __repr__(self):
        return f"{self.generator} \n\n {self.discriminator}"

    def real_loss(self, d_out: torch.Tensor) -> torch.Tensor:
        batch_size = d_out.size(0)
        labels = torch.ones(batch_size).to(
            self.device_info.backend) * 0.8  # todo: replace with smoothing
        return self.__calculate_loss(d_out, labels)

    def fake_loss(self, d_out: torch.Tensor) -> torch.Tensor:
        batch_size = d_out.size(0)
        labels = torch.ones(batch_size).to(
            self.device_info.backend) * 0.1  # todo: waht?
        return self.__calculate_loss(d_out, labels)

    def __calculate_loss(self, output, labels):
        criterion = nn.BCELoss()
        return criterion(output.squeeze(), labels)

    def noise(self, size):
        z = np.random.uniform(-1, 1, size=size)
        return torch.from_numpy(z).float().to(self.device_info.backend)

    def train_generator(self, optimizer, size) -> tuple[float, float]:
        optimizer.zero_grad()

        z = self.noise(size)
        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        g_loss = self.real_loss(d_fake)

        g_loss.backward()
        optimizer.step()

        # Calculate D(G(z)) for monitoring
        d_gz2 = d_fake.mean().item()

        return g_loss.item(), d_gz2

    def train_discriminator(self, optimizer, real_image, size) -> tuple[float, float, float, float, float]:
        optimizer.zero_grad()
        d_real = self.discriminator(real_image.to(
            self.device_info.backend)).view(-1)
        d_real_loss = self.real_loss(d_real)

        z = self.noise(size)
        fake_images = self.generator(z)

        d_fake = self.discriminator(fake_images.detach())
        d_fake_loss = self.fake_loss(d_fake)

        total_loss = d_real_loss + d_fake_loss

        total_loss.backward()
        optimizer.step()

        # Calculate additional metrics for monitoring
        d_x = d_real.mean().item()  # D(x) - discriminator output on real images
        # D(G(z)) - discriminator output on fake images during D training
        d_gz1 = d_fake.mean().item()

        # Calculate accuracies (assuming threshold of 0.5)
        d_real_acc = (d_real > 0.5).float().mean().item()
        d_fake_acc = (d_fake < 0.5).float().mean().item()

        return total_loss.item(), d_x, d_gz1, d_real_acc, d_fake_acc

    def optimizers(self) -> tuple[Adam, Adam]:
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            self.config.discriminator_lr,
            (self.config.discriminator_beta1, 0.999))
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            self.config.generator_lr,
            (self.config.generator_beta1, 0.999)
        )
        return d_optimizer, g_optimizer

    def train(self) -> tuple[list[Any], list[Any]] | None:
        samples, losses = [], []
        sample_size = 16
        z_size = self.config.latent_size

        d_optimizer, g_optimizer = self.optimizers()

        z = self.noise((sample_size, z_size))

        self.discriminator.train()
        self.generator.train()
        epochs = self.config.epochs

        global_step = 0  # Initialize global step counter

        # Track epoch losses for averaging
        epoch_d_losses, epoch_g_losses = [], []

        for epoch in range(epochs):
            epoch_d_losses.clear()
            epoch_g_losses.clear()

            for i, real_images in enumerate(self.dataset.dataloader):
                batch_size = real_images.size(0)

                d_loss, d_x, d_gz1, d_real_acc, d_fake_acc = self.train_discriminator(
                    d_optimizer, real_images, (sample_size, z_size))
                g_loss, d_gz2 = self.train_generator(
                    g_optimizer, (sample_size, z_size))

                epoch_d_losses.append(d_loss)
                epoch_g_losses.append(g_loss)

                if i % self.config.image_log_every_iters == 0:
                    logger.log(
                        f'Epoch [{epoch+1:4d}/{epochs:4d}] | d_loss {d_loss:6.4f} | g_loss {g_loss:6.4f} | D(x) {d_x:.4f} | D(G(z)) {d_gz1:.4f}/{d_gz2:.4f} | D_acc {d_real_acc:.3f}/{d_fake_acc:.3f}')

                    if self.wandb_logger:
                        self.wandb_logger.log_training_losses(
                            d_loss=d_loss,
                            g_loss=g_loss,
                            epoch=epoch + 1,
                            iteration=i,
                            d_x=d_x,
                            d_gz1=d_gz1,
                            d_gz2=d_gz2,
                            d_real_acc=d_real_acc,
                            d_fake_acc=d_fake_acc,
                            step=global_step
                        )

                        if torch.cuda.is_available():
                            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                            self.wandb_logger.log_system_metrics(
                                gpu_memory_allocated=gpu_memory_allocated,
                                gpu_memory_reserved=gpu_memory_reserved
                            )

                global_step += 1  # Increment global step for every iteration

            losses.append((d_loss, g_loss))

            self.generator.eval()
            with torch.no_grad():
                epoch_samples = self.generator(z)
                samples.append(epoch_samples)
            self.generator.train()

            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)

            if self.wandb_logger:
                self.wandb_logger.log_epoch_summary(
                    epoch=epoch + 1,
                    avg_d_loss=avg_d_loss,
                    avg_g_loss=avg_g_loss,
                    samples=epoch_samples[:8],  # Log first 8 samples
                    step=global_step
                )

                # Log image grids periodically
                if (epoch + 1) % 5 == 0:  # Every 5 epochs
                    # Get a batch of real images for comparison
                    real_batch = next(iter(self.dataset.dataloader))
                    self.wandb_logger.log_image_grids(
                        fake_samples=epoch_samples[:16],
                        real_samples=real_batch[:16],
                        step=global_step,
                        caption=f"Epoch {epoch + 1}"
                    )

                # Evaluate MiFID if configured
                if (self.config.mifid_eval_every_epochs > 0 and
                        (epoch + 1) % self.config.mifid_eval_every_epochs == 0):
                    mifid_score = self.wandb_logger.evaluate_mifid(
                        generator=self.generator,
                        dataloader=self.dataset.dataloader,
                        latent_size=self.config.latent_size,
                        device=self.device_info.backend,
                        num_batches=self.config.mifid_eval_batches,
                        step=global_step
                    )
                    if mifid_score is not None:
                        logger.log(
                            f"MiFID Score: {mifid_score:.4f}", color=logger.Colors.GREEN)

                # Increment global step after all epoch logging is complete
                global_step += 1

        # Save the final trained models
        self._save_final_models()

        return samples, losses

    def _save_final_models(self):
        """Save the final trained generator and discriminator models"""
        import os
        from pathlib import Path

        # Create artifacts directory if it doesn't exist
        Path(self.config.artifacts_folder).mkdir(parents=True, exist_ok=True)

        # Save generator
        generator_path = os.path.join(
            self.config.artifacts_folder, f"generator_{utils.get_run_name(self.wandb_logger)}.pth")
        torch.save(self.generator.state_dict(), generator_path)
        logger.log(
            f"✅ Generator saved to {generator_path}", color=logger.Colors.GREEN)

        # Save discriminator
        discriminator_path = os.path.join(
            self.config.artifacts_folder, f"discriminator_{utils.get_run_name(self.wandb_logger)}.pth")
        torch.save(self.discriminator.state_dict(), discriminator_path)
        logger.log(
            f"✅ Discriminator saved to {discriminator_path}", color=logger.Colors.GREEN)
