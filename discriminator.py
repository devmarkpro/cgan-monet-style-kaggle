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
            # 256 -> 128
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 64
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(in_channels=ndf * 8, out_channels=ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            nn.Conv2d(in_channels=ndf * 16, out_channels=ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 -> 1
            nn.Conv2d(in_channels=ndf * 16, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)