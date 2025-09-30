import torch
import torch.nn as nn

from device import get_device_info


class Discriminator(nn.Module):
    def __init__(self, num_channels: int, feature_map_size: int):
        super(Discriminator, self).__init__()
        self.ngpu = get_device_info().ngpu
        self.channels = num_channels
        self.feature_map_size = feature_map_size
        self._build_model()

    def _build_model(self):
        nc = self.channels
        ndf = self.feature_map_size

        self._forward = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)
