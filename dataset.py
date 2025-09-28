from typing import Optional
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from device import get_device_info


class Dataset:
    def __init__(
        self,
        data_dir: str,
        artifacts_folder: str,
        batch_size: int = 128,
        workers: int = 2,
    ):
        self.dir = data_dir
        self.image_size = 256
        self.batch_size = batch_size
        self.workers = workers
        self.device = get_device_info().device
        self._set_dataset()
        self.artifacts_folder = artifacts_folder

    def _set_dataset(self):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.dataset = dset.ImageFolder(root=self.dir, transform=transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return self.dataset, self.dataloader
