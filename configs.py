from dataclasses import dataclass
import logging


@dataclass
class AppParams:
    seed: int = 42
    log_level: int = logging.INFO
    dataset_dir: str = "./data/monet/training"
    batch_size: int = 128
    workers: int = 1
    epochs: int = 100
    artifacts_folder: str = "./artifacts"
    use_wandb: bool = False
    num_channels: int = 3
    latent_size: int = 100
    generator_feature_map_size: int = 64
    discriminator_feature_map_size: int = 64
    learning_rate: float = 0.0002
    discriminator_beta1: float = 0.999
    generator_beta1: float = 0.999
