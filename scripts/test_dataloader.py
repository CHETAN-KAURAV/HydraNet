from torch.utils.data import DataLoader

from utils.dataset import FloodDataset

# -----------------------------
# Dataset Paths
# -----------------------------
IMAGE_DIR = "../data/images"
MASK_DIR = "../data/masks"

# -----------------------------
# Create Dataset
# -----------------------------
dataset = FloodDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=256
)

# -----------------------------
# Create DataLoader
# -----------------------------
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# -----------------------------
# Load One Batch
# -----------------------------
images, masks = next(iter(dataloader))

# -----------------------------
# Print Batch Info
# -----------------------------
print("Images Shape:", images.shape)
print("Masks Shape :", masks.shape)

print("\nImage dtype:", images.dtype)
print("Mask dtype :", masks.dtype)

print("\nMask Unique Values:")
print(masks.unique())