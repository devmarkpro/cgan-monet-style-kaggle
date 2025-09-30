import torch
import torch.nn as nn

from device import get_device_info

BATCH_SIZE = 16
N_WORKERS = 0

Z_SIZE = 128
SAMPLE_SIZE = 16
CONV_DIM = 64

lr = 0.0002
beta1 = 0.5
beta2 = 0.999


class Generator(nn.Module):
    def __init__(self, num_channels: int, latent_size: int, feature_map_size: int):
        super(Generator, self).__init__()

        self.num_channels = num_channels
        self.latent_size = latent_size
        self.generator_feature_map_size = feature_map_size

        self.device_info = get_device_info()
        self._build_model()

    def _build_model(self):
        # Number of channels in the training images. For color images this is 3
        nc = self.num_channels

        # Size of z latent vector (i.e. size of generator input)
        nz = self.latent_size

        # Size of feature maps in generator
        ngf = self.generator_feature_map_size

        self._forward = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self._forward(x)
