from typing import Any, Optional

import torch
from torch import nn, optim
from torch.optim import Adam

import device
import logger
import utils
from configs import AppParams
from dataset import Dataset
import torchvision.utils as vutils
import wandb
from torch.utils.data import DataLoader


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
            self.wandb_logger.log_model_gradients(self.generator, self.discriminator)
    def __str__(self):
        return f"{self.generator} \n\n {self.discriminator}"
    def __repr__(self):
        return f"{self.generator} \n\n {self.discriminator}"

    def real_loss(self, d_out: torch.Tensor) -> torch.Tensor:
        batch_size = d_out.size(0)
        labels = torch.ones(batch_size).to(self.device_info.backend) * 0.8 # todo: replace with smoothing
        return self.__calculate_loss(d_out, labels)

    def fake_loss(self, d_out: torch.Tensor) -> torch.Tensor:
        batch_size = d_out.size(0)
        labels = torch.ones(batch_size).to(self.device_info.backend) * 0.1 # todo: waht?
        return self.__calculate_loss(d_out, labels)

    def __calculate_loss(self, output, labels):
        criterion = nn.BCELoss()
        return criterion(output.squeeze(), labels)


    def noise(self, size):
        # Match the working notebook: use uniform distribution
        import numpy as np
        z = np.random.uniform(-1, 1, size=size)
        return torch.from_numpy(z).float().to(self.device_info.backend)

    def train_generator(self, optimizer, size) -> float:
        optimizer.zero_grad()

        z = self.noise(size)
        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        g_loss = self.real_loss(d_fake)

        g_loss.backward()
        optimizer.step()
        
        return g_loss.item()
    def train_discriminator(self, optimizer, real_image, size) -> float:
        optimizer.zero_grad()
        d_real = self.discriminator(real_image.to(self.device_info.backend)).view(-1)
        d_real_loss = self.real_loss(d_real)

        z = self.noise(size)
        fake_images = self.generator(z)

        # CRITICAL: Match notebook - NO .detach() and NO .view(-1)!
        d_fake = self.discriminator(fake_images)
        # d_fake = self.discriminator(fake_images.detach())
        d_fake_loss = self.fake_loss(d_fake)

        total_loss = d_real_loss + d_fake_loss

        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
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

        for epoch in range(epochs):
            for i, real_images in enumerate(self.dataset.dataloader):
                batch_size = real_images.size(0)
                
                # Train discriminator and generator (match notebook approach)
                d_loss = self.train_discriminator(d_optimizer, real_images, (sample_size, z_size))
                g_loss = self.train_generator(g_optimizer, (sample_size, z_size))

                if i % self.config.image_log_every_iters == 0:
                    logger.log(f'Epoch [{epoch+1:4d}/{epochs:4d}] | d_loss {d_loss:6.4f} | g_loss {g_loss:6.4f}')
                
                global_step += 1  # Increment global step for every iteration

            # Record losses exactly like notebook (last batch of epoch)
            losses.append((d_loss, g_loss))

            # Generate samples for this epoch (match notebook approach)
            self.generator.eval()
            samples.append(self.generator(z))
            self.generator.train()
            
        return samples, losses