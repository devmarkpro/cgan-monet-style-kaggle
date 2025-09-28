import torch
import torch.nn as nn

from device import get_device_info


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


        # For 256x256 output: 1->4->8->16->32->64->128->256
        self._forward = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=ngf * 16),
            nn.ReLU(inplace=True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(in_channels=ngf * 16, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=ngf // 2),
            nn.ReLU(inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(in_channels=ngf // 2, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)