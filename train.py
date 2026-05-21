import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.dataset import FloodDataset
from models.unet_model import build_unet
from utils.losses import BCEDiceLoss

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", DEVICE)

# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# -----------------------------
# Hyperparameters
# -----------------------------
IMAGE_SIZE = 256
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 5

# -----------------------------
# Dataset
# -----------------------------
dataset = FloodDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)

# -----------------------------
# Train / Validation Split
# -----------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------
model = build_unet().to(DEVICE)

# -----------------------------
# Loss Function
# -----------------------------
criterion = BCEDiceLoss()

# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # -------------------------
    # TRAINING
    # -------------------------
    model.train()

    train_loss = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, masks in train_bar:

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, masks)

        # Backprop
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix(
            loss=loss.item()
        )

    avg_train_loss = train_loss / len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # -------------------------
    # Print Metrics
    # -------------------------
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss  : {avg_val_loss:.4f}")

    # -------------------------
    # Save Best Model
    # -------------------------
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved!")