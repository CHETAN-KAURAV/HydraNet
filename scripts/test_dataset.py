from utils.dataset import FloodDataset

# Dataset paths
IMAGE_DIR = "../data/images"
MASK_DIR = "../data/masks"

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