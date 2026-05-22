import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from models.unet_model import build_unet


# DEVICE


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {DEVICE}")


# PATHS


IMAGE_DIR = "/content/drive/MyDrive/HydraNetData/images"
MASK_DIR = "/content/drive/MyDrive/HydraNetData/masks"

MODEL_PATH = "best_model.pth"

IMAGE_SIZE = 256


# LOAD MODEL


model = build_unet()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)

model.eval()

print("Model Loaded Successfully!")


# SELECT RANDOM SAMPLE


image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".tif")
])

sample_image = random.choice(image_files)

sample_mask = sample_image.replace(
    "S1Hand",
    "LabelHand"
)

image_path = os.path.join(
    IMAGE_DIR,
    sample_image
)

mask_path = os.path.join(
    MASK_DIR,
    sample_mask
)

print(f"\nSelected Image: {sample_image}")


# LOAD IMAGE


with rasterio.open(image_path) as src:
    image = src.read(1).astype(np.float32)


# LOAD MASK


with rasterio.open(mask_path) as src:
    mask = src.read(1).astype(np.float32)


# CLEAN IMAGE


image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# NORMALIZATION


mean = image.mean()
std = image.std()

if std < 1e-6:
    std = 1e-6

image_norm = (image - mean) / std


# RESIZE


image_resized = cv2.resize(
    image_norm,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_LINEAR
)

mask_resized = cv2.resize(
    mask,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_NEAREST
)


# PREPARE TENSOR


input_tensor = np.expand_dims(
    image_resized,
    axis=0
)

input_tensor = np.expand_dims(
    input_tensor,
    axis=0
)

input_tensor = input_tensor.clone().detach()

input_tensor = input_tensor.to(
    DEVICE,
    dtype=torch.float32
)


# INFERENCE


with torch.no_grad():

    output = model(input_tensor)

    prediction = torch.sigmoid(output)


# CONVERT TO NUMPY


prediction = prediction.squeeze().cpu().numpy()


# THRESHOLD


pred_mask = (prediction > 0.3).astype(np.uint8)


# VISUALIZATION NORMALIZATION


image_vis = (image_resized - image_resized.min()) / (
        image_resized.max() - image_resized.min()
)


# VISUALIZATION


fig, ax = plt.subplots(1, 3, figsize=(18, 6))


# SAR IMAGE


ax[0].imshow(image_vis, cmap="gray")

ax[0].set_title("SAR Image")

ax[0].axis("off")


# GROUND TRUTH


ax[1].imshow(mask_resized, cmap="gray")

ax[1].set_title("Ground Truth Mask")

ax[1].axis("off")


# PREDICTION


# PREDICTION PROBABILITY MAP

ax[2].imshow(prediction, cmap="jet")

ax[2].set_title("Flood Probability Map")

ax[2].axis("off")

plt.tight_layout()

plt.savefig("prediction_result.png")
print("Visualization saved!")