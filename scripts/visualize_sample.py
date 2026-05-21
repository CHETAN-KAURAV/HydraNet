import os
import random

import rasterio
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Dataset Paths
# -----------------------------
IMAGE_DIR = r"C:\Users\cheta\PycharmProjects\HydraNet\data\images"
MASK_DIR = r"C:\Users\cheta\PycharmProjects\HydraNet\data\masks"

# -----------------------------
# Get all image files
# -----------------------------
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".tif")
])

# -----------------------------
# Pick one random sample
# -----------------------------
sample_image = random.choice(image_files)

# Convert image filename to mask filename
sample_mask = sample_image.replace("S1Hand", "LabelHand")

image_path = os.path.join(IMAGE_DIR, sample_image)
mask_path = os.path.join(MASK_DIR, sample_mask)

print(f"\nSelected Image : {sample_image}")
print(f"Selected Mask  : {sample_mask}")

# -----------------------------
# Load SAR Image
# -----------------------------
with rasterio.open(image_path) as src:
    image = src.read(1)

# -----------------------------
# Load Flood Mask
# -----------------------------
with rasterio.open(mask_path) as src:
    mask = src.read(1)

# -----------------------------
# Print Information
# -----------------------------
print("\nImage Shape:", image.shape)
print("Mask Shape :", mask.shape)

print("\nSAR Image Statistics")
print("---------------------")
print("Min :", np.min(image))
print("Max :", np.max(image))
print("Mean:", np.mean(image))

print("\nMask Unique Values:", np.unique(mask))

# -----------------------------
# Normalize SAR Image for Visualization
# -----------------------------
image_vis = (image - image.min()) / (image.max() - image.min())

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image_vis, cmap="gray")
ax[0].set_title("Sentinel-1 SAR Image")
ax[0].axis("off")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Flood Mask")
ax[1].axis("off")

plt.tight_layout()
plt.show()