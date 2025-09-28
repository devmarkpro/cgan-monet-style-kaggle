import logging
import os

import random

import torch
import numpy as np
from torch import nn, optim

from configs import AppParams
import utils
from dataset import Dataset
import device
import logger
from discriminator import Discriminator
from generator import Generator
from monet_wandb import MonetWandb


import torchvision.utils as vutils


class App(object):
    def __init__(self, params: AppParams):
        self.configs = params
        print("Setting up environment...")
        print(f"log level: {params.log_level} | seed: {params.seed}")

        logger.setup()
        self.device_info = device.get_device_info()
        logger.log(f"Device info: {self.device_info}")

        self._set_random_seed()
        self._set_dataset()
        if self.configs.use_wandb:
            self._setup_wandb()
        else:
            logger.log(
                "Wandb has not setup, set --use_wandb 1 to enable using it",
                level=logging.WARNING,
                color=logger.Colors.YELLOW,
            )

        logger.log("Setting up model...")
        self._build_generator()
        self._build_discriminator()

    def __call__(self, *args, **kwargs):
        self.run()

    def run(self):
        logger.log("starting the experiment")

        logger.log(self.generator, level=logging.DEBUG)
        logger.log(self.discriminator, level=logging.DEBUG)
        return self._train()

    def _set_random_seed(self):
        random.seed(self.configs.seed)
        np.random.seed(self.configs.seed)
        torch.manual_seed(self.configs.seed)

    def _set_dataset(self):
        self.ds = Dataset(
            data_dir=self.configs.dataset_dir,
            batch_size=self.configs.batch_size,
            workers=self.configs.workers,
            artifacts_folder=self.configs.artifacts_folder,
        )

    def _setup_wandb(self):
        self.wandb = MonetWandb(
            project_name=os.getenv("WANDB_PROJECT_NAME", "dcgan-monet"),
            params=self.configs,
        )

    def _build_generator(self):
        self.generator = Generator(
            num_channels=self.configs.num_channels,
            latent_size=self.configs.latent_size,
            feature_map_size=self.configs.generator_feature_map_size,
        ).to(self.device_info.device)
        if self.device_info.device == "gpu" or self.device_info.device == "mps":
            if self.device_info.ngpu > 1:
                self.generator = nn.DataParallel(
                    self.generator, list(range(self.device_info.ngpu))
                )
        self.generator.apply(utils.weights_init)

    def _build_discriminator(self):
        self.discriminator = Discriminator(
            num_channels=self.configs.num_channels,
            feature_map_size=self.configs.discriminator_feature_map_size,
        ).to(self.device_info.device)

        if self.device_info.device == "gpu" or self.device_info.device == "mps":
            if self.device_info.ngpu > 1:
                self.discriminator = nn.DataParallel(
                    self.discriminator, list(range(self.device_info.ngpu))
                )
        self.discriminator.apply(utils.weights_init)

    def _train(self) -> tuple:
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(
            64, self.configs.latent_size, 1, 1, device=self.device_info.device
        )

        # Establish convention for real and fake labels during training
        real_label = 1.0
        fake_label = 0.0

        # Learning rate for optimizers
        lr = self.configs.learning_rate

        # Beta1 hyperparameter for Adam optimizers
        discriminator_beta1 = self.configs.discriminator_beta1
        generator_beta1 = self.configs.generator_beta1

        # Setup Adam optimizers for both G and D
        discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(discriminator_beta1, 0.999)
        )
        generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(generator_beta1, 0.999)
        )

        # Lists to keep track of progress
        img_list = []
        generator_losses = []
        discriminator_losses = []
        iters = 0
        num_epochs = self.configs.epochs
        logger.log("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.ds.dataloader, 0):
                self.discriminator.zero_grad()

                real_cpu = data[0].to(self.device_info.device)

                output = self.discriminator(real_cpu).view(-1)
                b_size = output.shape[0]

                label = torch.full(
                    size=(b_size,),
                    fill_value=real_label,
                    device=self.device_info.device,
                )

                discriminator_error_real = criterion(output, label)
                discriminator_error_real.backward()
                d_x = output.mean().item()

                noise = torch.randn(
                    b_size,
                    self.configs.latent_size,
                    1,
                    1,
                    device=self.device_info.device,
                )
                fake = self.generator(noise)
                label.fill_(fake_label)
                output = self.discriminator(fake.detach()).view(-1)
                discriminator_error_fake = criterion(output, label)
                discriminator_error_fake.backward()
                d_g_z1 = output.mean().item()
                error_discriminator = (
                    discriminator_error_real + discriminator_error_fake
                )
                discriminator_optimizer.step()

                # Update G
                self.generator.zero_grad()
                label.fill_(real_label)
                output = self.discriminator(fake).view(-1)
                error_generator = criterion(output, label)
                error_generator.backward()
                d_g_z2 = output.mean().item()
                generator_optimizer.step()

                if self.configs.use_wandb:
                    self.wandb.run.log(
                        {
                            "Discriminator Loss": error_discriminator.item(),
                            "Generator Loss": error_generator.item(),
                            "D(x)": d_x,
                            "D(G(z))1": d_g_z1,
                            "D(G(z))2": d_g_z2,
                            "Epoch": epoch,
                            "Batch": i,
                        }
                    )
                # Log every 5 iterations or every batch if few batches
                if (iters % 5 == 0) or (len(self.ds.dataloader) <= 5):
                    logger.log(
                        f"[epoch {epoch}/{num_epochs}] [{i}/{len(self.ds.dataloader)}]: "
                        f"loss_discriminator: {error_discriminator.item():.4f} "
                        f"loss_generator: {error_generator.item():.4f} "
                        f"D(x): {d_x:.4f} "
                        f"D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}",
                        color=logger.Colors.CYAN,
                    )
                generator_losses.append(error_generator.item())
                discriminator_losses.append(discriminator_error_fake.item())
                if (iters % 10 == 0) or (
                    (epoch == num_epochs - 1) and (i == len(self.ds.dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    logger.log(f"Saving image at iter {iters}")
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1
        # save arrays for plotting later
        os.makedirs(self.configs.artifacts_folder, exist_ok=True)
        np.save(
            os.path.join(self.configs.artifacts_folder, "generator_losses.npy"),
            np.array(generator_losses),
        )
        np.save(
            os.path.join(self.configs.artifacts_folder, "discriminator_losses.npy"),
            np.array(discriminator_losses),
        )
        np.save(
            os.path.join(self.configs.artifacts_folder, "img_list.npy"),
            np.array([img.numpy() for img in img_list]),
        )
        return img_list, generator_losses, discriminator_losses
