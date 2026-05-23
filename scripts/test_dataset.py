from utils.dataset import FloodDataset

# Dataset paths
IMAGE_DIR = "/content/drive/MyDrive/HydraNetData/images"
MASK_DIR = "/content/drive/MyDrive/HydraNetData/masks"

# Create dataset
dataset = FloodDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=256
)

print("Dataset Size:", len(dataset))

# Load one sample
image, mask = dataset[0]

print("\nImage Shape:", image.shape)
print("Mask Shape :", mask.shape)

print("\nImage Type:", image.dtype)
print("Mask Type :", mask.dtype)

print("\nMask Unique Values:")
print(mask.unique())