import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, 
                    root_path: str, 
                    image_path: str, 
                    heatmap_path: str, 
                    mask_path: str,
                    image_size: int,
                    subsample: int = 1.0):
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.heatmaps = sorted([root_path+f"/{heatmap_path}/"+i for i in os.listdir(root_path+f"/{heatmap_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])
        
        if len(self.images) != len(self.masks) and len(self.masks) != len(self.heatmaps): 
            raise ValueError("Length of images, masks, and heatmaps are not the same")
        
        # Subsample
        self.images = self.images[:int(len(self.images)*subsample)]
        self.heatmaps = self.heatmaps[:int(len(self.heatmaps)*subsample)]
        self.masks = self.masks[:int(len(self.masks)*subsample)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_size = image_size

        
    def __getitem__(self, index) -> torch.tensor:
        img = Image.open(self.images[index]).convert("RGBA")
        heatmap = Image.open(self.heatmaps[index]).convert("L")        
        mask = Image.open(self.masks[index]).convert("L")

        # Transform all images
        img_tensor = self.transform(img)
        heatmap_tensor = self.transform(heatmap)
        mask_tensor = self.transform(mask)

        # Heatmap normalization
        mean = heatmap_tensor.mean()
        std = heatmap_tensor.std()

        if std > 0: heatmap_tensor = (heatmap_tensor - mean) / std
        else: heatmap_tensor = heatmap_tensor - mean # Avoid NaN if std = 0

        # Optional: re-scale to [0, 1] again if needed
        # This keeps intensity relative to overall heatmap contrast but normalized distribution
        heatmap_tensor = (heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-8)

        # Stack img and heatmap along channel dimensions
        concat_tensor = torch.cat([img_tensor, heatmap_tensor], dim=0) # Shape (5, H, W)
        
        return concat_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.images)
