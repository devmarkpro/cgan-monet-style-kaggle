import logging
import os

import random
import time

import torch
import wandb
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.image.mifid import (
    MemorizationInformedFrechetInceptionDistance as MiFID,
)

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
        logger.log(self.configs, level=logging.DEBUG)
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
        criterion = nn.BCEWithLogitsLoss()
        # Optimize convolution autotuning on CUDA
        if self.device_info.backend == "cuda":
            torch.backends.cudnn.benchmark = True

        # Attach gradient watchers once
        if self.configs.use_wandb:
            try:
                wandb.watch(self.generator, log="gradients", log_freq=100)
                wandb.watch(self.discriminator, log="gradients", log_freq=100)
            except Exception as e:
                logger.log(
                    f"wandb.watch failed: {e}",
                    level=logging.ERROR,
                    color=logger.Colors.RED,
                )

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(
            64, self.configs.latent_size, 1, 1, device=self.device_info.device
        )

        # Establish convention for real and fake labels during training
        real_label = 1.0  # generator target stays 1.0
        fake_label = 0.0
        # One-sided label smoothing applied only to D(real)
        real_label_smooth = float(self.configs.label_smoothing_real)

        # Learning rate for optimizers
        discriminator_lr = self.configs.discriminator_lr
        generator_lr = self.configs.generator_lr

        # Beta1 hyperparameter for Adam optimizers
        discriminator_beta1 = self.configs.discriminator_beta1
        generator_beta1 = self.configs.generator_beta1

        # Setup Adam optimizers for both G and D
        discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr, betas=(discriminator_beta1, 0.999)
        )
        generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=generator_lr, betas=(generator_beta1, 0.999)
        )
        discriminator_scheduler = LinearLR(
            discriminator_optimizer,
            start_factor=1.0,
            end_factor=0.3,
            total_iters=self.configs.epochs,
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
                batch_t0 = time.time()
                self.discriminator.zero_grad()

                # Pinned memory + non_blocking speeds up H2D transfer
                real_cpu = data[0].to(self.device_info.device, non_blocking=(self.device_info.backend == "cuda"))

                output_real = self.discriminator(real_cpu).view(-1)
                b_size = output_real.shape[0]

                label = torch.full(
                    size=(b_size,),
                    fill_value=real_label_smooth,
                    device=self.device_info.device,
                )

                discriminator_error_real = criterion(output_real, label)
                discriminator_error_real.backward()
                d_x = torch.sigmoid(output_real).mean().item()
                # Discriminator accuracy on real (threshold at 0 logit -> 0.5 prob)
                d_real_acc = (output_real.detach() >= 0.0).float().mean().item()

                noise = torch.randn(
                    b_size,
                    self.configs.latent_size,
                    1,
                    1,
                    device=self.device_info.device,
                )
                fake = self.generator(noise)
                label.fill_(fake_label)
                output_fake = self.discriminator(fake.detach()).view(-1)
                discriminator_error_fake = criterion(output_fake, label)
                discriminator_error_fake.backward()
                d_g_z1 = torch.sigmoid(output_fake).mean().item()
                # Discriminator accuracy on fake (threshold at 0 logit -> 0.5 prob)
                d_fake_acc = (output_fake.detach() < 0.0).float().mean().item()
                error_discriminator = (
                    discriminator_error_real + discriminator_error_fake
                )
                # Grad norm before step
                grad_norm_D = utils.grad_l2_norm(self.discriminator)
                discriminator_optimizer.step()

                # Update G (freeze D to avoid computing grads for D)
                self.generator.zero_grad()
                for p in self.discriminator.parameters():
                    p.requires_grad_(False)
                label.fill_(real_label)
                output = self.discriminator(fake).view(-1)
                error_generator = criterion(output, label)
                error_generator.backward()
                d_g_z2 = torch.sigmoid(output).mean().item()
                grad_norm_G = utils.grad_l2_norm(self.generator)
                generator_optimizer.step()
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)

                # Timing and system metrics
                batch_dt = max(time.time() - batch_t0, 1e-6)
                batch_time_ms = batch_dt * 1000.0
                imgs_per_sec = float(b_size) / batch_dt
                lr_D = discriminator_optimizer.param_groups[0]["lr"]
                lr_G = generator_optimizer.param_groups[0]["lr"]

                gpu_mem_alloc_mb, gpu_mem_reserved_mb = self._get_gpu_memory_info()
                

                if self.configs.use_wandb:
                    log_payload = {
                        # legacy scalar names
                        "Discriminator Loss": error_discriminator.item(),
                        "Generator Loss": error_generator.item(),
                        "D(x)": d_x,
                        "D(G(z))1": d_g_z1,
                        "D(G(z))2": d_g_z2,
                        "Epoch": epoch,
                        "Batch": i,
                        # structured scalars
                        "train/loss/D_total": error_discriminator.item(),
                        "train/loss/D_real": discriminator_error_real.item(),
                        "train/loss/D_fake": discriminator_error_fake.item(),
                        "train/loss/G": error_generator.item(),
                        "train/Dx": d_x,
                        "train/DGz1": d_g_z1,
                        "train/DGz2": d_g_z2,
                        "train/acc/D_real": d_real_acc,
                        "train/acc/D_fake": d_fake_acc,
                        "train/acc/D": 0.5 * (d_real_acc + d_fake_acc),
                        "train/grad_norm/D": grad_norm_D,
                        "train/grad_norm/G": grad_norm_G,
                        "opt/lr/D": lr_D,
                        "opt/lr/G": lr_G,
                        "time/batch_ms": batch_time_ms,
                        "time/imgs_per_sec": imgs_per_sec,
                    }
                    if gpu_mem_alloc_mb is not None:
                        log_payload["sys/gpu/mem_alloc_mb"] = gpu_mem_alloc_mb
                    if gpu_mem_reserved_mb is not None:
                        log_payload["sys/gpu/mem_reserved_mb"] = gpu_mem_reserved_mb
                    self.wandb.run.log(log_payload, step=iters)
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
                # End-of-epoch evaluation using MiFID
                eval_every = self.configs.mifid_eval_every_epochs
                eval_batches_cfg = self.configs.mifid_eval_batches
                should_eval = (
                    self.configs.use_wandb
                    and (i == len(self.ds.dataloader) - 1)
                    and (eval_every > 0)
                    and ((epoch + 1) % eval_every == 0)
                )
                if should_eval:
                    try:
                        num_eval_batches = min(eval_batches_cfg, len(self.ds.dataloader))
                        mifid = MiFID(feature=2048, normalize=False, reset_real_features=False)
                        with torch.no_grad():
                            for j, data_eval in enumerate(self.ds.dataloader, 0):
                                if j >= num_eval_batches:
                                    break
                                real_b = data_eval[0]
                                real_u8 = utils.denorm_to_uint8(real_b).cpu()
                                mifid.update(real_u8, real=True)
                                z = torch.randn(
                                    real_b.size(0),
                                    self.configs.latent_size,
                                    1,
                                    1,
                                    device=self.device_info.device,
                                )
                                fake_b = self.generator(z).detach().cpu()
                                fake_u8 = utils.denorm_to_uint8(fake_b)
                                mifid.update(fake_u8, real=False)
                        mifid_score = float(mifid.compute().item())
                        logger.log(f"MiFID score: {mifid_score} | num batches: {num_eval_batches} | epoch: {epoch}", color=logger.Colors.GREEN)
                        self.wandb.run.log(
                            {
                                "eval/mifid": mifid_score,
                                "eval/num_batches": num_eval_batches,
                                "Epoch": epoch,
                            },
                            step=iters,
                        )
                    except Exception as e:
                        logger.log(
                            f"MiFID evaluation failed: {e}",
                            level=logging.WARNING,
                            color=logger.Colors.YELLOW,
                        )
                # Step schedulers once per epoch (after last batch)
                if i == len(self.ds.dataloader) - 1:
                    discriminator_scheduler.step()
                image_log_every = self.configs.image_log_every_iters
                if (iters % image_log_every == 0) or (
                    (epoch == num_epochs - 1) and (i == len(self.ds.dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    logger.log(f"Saving image at iter {iters}")
                    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    img_list.append(img_grid)
                    if self.configs.use_wandb:
                        try:
                            self.wandb.run.log(
                                {
                                    "samples/fake_grid": wandb.Image(
                                        img_grid, caption=f"iter {iters}"
                                    )
                                },
                                step=iters,
                            )
                            # also log a real grid for comparison
                            real_grid = vutils.make_grid(
                                real_cpu[: min(64, real_cpu.size(0))].detach().cpu(),
                                padding=2,
                                normalize=True,
                            )
                            self.wandb.run.log(
                                {"samples/real_grid": wandb.Image(real_grid)},
                                step=iters,
                            )
                        except Exception as e:
                            logger.log(
                                f"wandb image log failed: {e}",
                                level=logging.WARNING,
                                color=logger.Colors.YELLOW,
                            )
                # advance global step
                iters += 1
        # save arrays for plotting later
        
        os.makedirs(self.configs.artifacts_folder, exist_ok=True)
        np.save(
            os.path.join(self.configs.artifacts_folder, f"generator_losses_{self.wandb.run.name}.npy"),
            np.array(generator_losses),
        )
        np.save(
            os.path.join(self.configs.artifacts_folder, f"discriminator_losses_{self.wandb.run.name}.npy"),
            np.array(discriminator_losses),
        )
        np.save(
            os.path.join(self.configs.artifacts_folder, f"img_list_{self.wandb.run.name}.npy"),
            np.array([img.numpy() for img in img_list]),
        )
        return img_list, generator_losses, discriminator_losses

    def _get_gpu_memory_info(self):
        gpu_mem_alloc_mb = None
        gpu_mem_reserved_mb = None

        if self.device_info.backend == "cuda":
            try:
                gpu_mem_alloc_mb = torch.cuda.memory_allocated() / 1e6
                gpu_mem_reserved_mb = torch.cuda.memory_reserved() / 1e6
            except Exception:
                gpu_mem_alloc_mb = None
                gpu_mem_reserved_mb = None
        elif self.device_info.backend == "mps":
            try:
                gpu_mem_alloc_mb = torch.mps.current_allocated_memory() / 1e6
                gpu_mem_reserved_mb = torch.mps.driver_allocated_memory() / 1e6
            except Exception:
                gpu_mem_alloc_mb = None
                gpu_mem_reserved_mb = None
        if random.random() < 0.0001:
            logger.log(f"GPU memory allocated: {gpu_mem_alloc_mb:.2f} MB | reserved: {gpu_mem_reserved_mb:.2f} MB", color=logger.Colors.PURPLE, level=logging.DEBUG)
        return gpu_mem_alloc_mb, gpu_mem_reserved_mb