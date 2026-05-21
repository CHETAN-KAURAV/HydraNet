import os

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class FloodDataset(Dataset):

    def __init__(self,
                 image_dir,
                 mask_dir,
                 image_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # Get all SAR image filenames
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".tif")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        # -----------------------------
        # Get filenames
        # -----------------------------
        image_name = self.image_files[idx]

        mask_name = image_name.replace(
            "S1Hand",
            "LabelHand"
        )

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        # -----------------------------
        # Load SAR image
        # -----------------------------
        with rasterio.open(image_path) as src:
            image = src.read(1).astype(np.float32)

        # -----------------------------
        # Load flood mask
        # -----------------------------
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        # -----------------------------
        # Normalize SAR image
        # -----------------------------
        # Clean invalid values
        # -----------------------------
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # -----------------------------
        # Safe normalization
        # -----------------------------
        mean = image.mean()
        std = image.std()

        if std < 1e-6:
            std = 1e-6

        image = (image - mean) / std

        # -----------------------------
        # Handle invalid labels
        # -----------------------------
        # Convert -1 to 255 (ignore index)
        mask[mask == -1] = 255

        # -----------------------------
        # Resize image and mask
        # -----------------------------
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # -----------------------------
        # Add channel dimension
        # -----------------------------
        image = np.expand_dims(image, axis=0)

        # Shape:
        # (1, H, W)

        # -----------------------------
        # Convert to tensors
        # -----------------------------
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return image, mask