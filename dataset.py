import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
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
        self._set_dataset()
        self.artifacts_folder = artifacts_folder

    def _set_dataset(self):
        backend = get_device_info().backend
        pin_mem = True if backend == "cuda" else False
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
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
            drop_last=True,
            pin_memory=pin_mem,
            persistent_workers=True if self.workers > 0 else False,
        )
        return self.dataset, self.dataloader
