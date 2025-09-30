from dataclasses import dataclass
import logging


@dataclass
class AppParams:
    seed: int = 42
    log_level: int = logging.INFO
    dataset_dir: str = "./data/monet/training"
    batch_size: int = 128
    workers: int = 0
    epochs: int = 100
    artifacts_folder: str = "./artifacts"
    use_wandb: bool = False
    num_channels: int = 3
    latent_size: int = 100
    generator_feature_map_size: int = 64
    discriminator_feature_map_size: int = 64
    discriminator_lr: float = 0.0002
    generator_lr: float = 0.0002
    discriminator_beta1: float = 0.5
    generator_beta1: float = 0.5
    label_smoothing_real: float = 1.0
    mifid_eval_every_epochs: int = 1
    mifid_eval_batches: int = 5
    image_log_every_iters: int = 10
