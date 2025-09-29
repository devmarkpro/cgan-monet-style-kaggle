import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset, DataLoader
import os
from PIL import Image


class Dataset(TorchDataset):
    def __init__(
        self,
        img_dir: str,
        batch_size: int = 16,  # Match notebook default
        workers: int = 0,      # Match notebook default
    ):
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)

        # Filter only image files (match notebook behavior)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.img_list = [
            os.path.join(abspath, path) 
            for path in path_list 
            if os.path.splitext(path.lower())[1] in image_extensions
        ]

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # Create DataLoader once (like notebook)
        self.dataloader = DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=workers,
            pin_memory=True if workers > 0 else False,
            drop_last=True  # Ensure consistent batch sizes
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        path = self.img_list[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)
