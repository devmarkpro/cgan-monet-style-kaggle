import logging
import os

import random
import time

import torch
import numpy as np
from torch import nn

from configs import AppParams
import utils
from dataset import Dataset
import device
import logger
from dcgan import DCGAN
from discriminator import Discriminator
from generator import Generator
from monet_wandb import MonetWandb


import matplotlib.pyplot as plt


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
        logger.log(self.configs, level=logging.DEBUG)
        logger.log(self.generator, level=logging.DEBUG)
        logger.log(self.discriminator, level=logging.DEBUG)

        # Pass wandb logger to DCGAN if available
        wandb_logger = self.wandb if self.configs.use_wandb else None

        dcgan = DCGAN(
            generator=self.generator,
            discriminator=self.discriminator,
            configs=self.configs,
            dataset=self.ds,
            wandb_logger=wandb_logger,
        )
        samples, losses = dcgan.train()
        run_name = utils.get_run_name(wandb_logger)

        logger.log(f"Saving samples")
        samples_path = f"./artifacts/samples_{run_name}.npy"
        losses_path = f"./artifacts/losses_{run_name}.npy"
        # Convert tensors to CPU before saving as numpy arrays
        samples_cpu = [sample.cpu().detach() for sample in samples]
        np.save(samples_path, np.array(samples_cpu))
        np.save(losses_path, np.array(losses))
        logger.log(f"Samples saved to {samples_path}",
                   color=logger.Colors.GREEN)
        logger.log(f"Losses saved to {losses_path}", color=logger.Colors.GREEN)

        # Finish wandb run if it was used
        if wandb_logger:
            wandb_logger.finish()
        fig, axes = plt.subplots(
            figsize=(15, 10), nrows=2, ncols=4, sharey=True, sharex=True
        )
        for ax, img in zip(axes.flatten(), samples[-1]):
            _, w, h = img.size()

            img = img.detach().cpu().numpy()

            img = np.transpose(img, (1, 2, 0))

            img = ((img + 1) * 255 / (2)).astype(np.uint8)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            im = ax.imshow(img.reshape((w, h, 3)))

        plt.savefig(f"./artifacts/samples_{run_name}.png")
        plt.close()

    def _set_random_seed(self):
        random.seed(self.configs.seed)
        np.random.seed(self.configs.seed)
        torch.manual_seed(self.configs.seed)

    def _set_dataset(self):
        self.ds = Dataset(
            img_dir=self.configs.dataset_dir,
            batch_size=self.configs.batch_size,
            workers=self.configs.workers,
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
