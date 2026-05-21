import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.dataset import FloodDataset
from models.unet_model import build_unet
from utils.losses import BCEDiceLoss

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {DEVICE}")

# =====================================================
# DATA PATHS
# =====================================================

IMAGE_DIR = "/content/drive/MyDrive/HydraNetData/images"
MASK_DIR = "/content/drive/MyDrive/HydraNetData/masks"

# =====================================================
# HYPERPARAMETERS
# =====================================================

IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5

# =====================================================
# DATASET
# =====================================================

dataset = FloodDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_size=IMAGE_SIZE
)

print(f"\nTotal Dataset Size: {len(dataset)}")

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print(f"Training Samples  : {len(train_dataset)}")
print(f"Validation Samples: {len(val_dataset)}")

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# =====================================================
# MODEL
# =====================================================

model = build_unet().to(DEVICE)

# =====================================================
# LOSS FUNCTION
# =====================================================

criterion = BCEDiceLoss()

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# TRAINING LOOP
# =====================================================

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # =================================================
    # TRAINING
    # =================================================

    model.train()

    train_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for batch_idx, (images, masks) in enumerate(train_bar):

        # ---------------------------------------------
        # Move to GPU
        # ---------------------------------------------

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # ---------------------------------------------
        # CHECK INPUTS FOR NaNs
        # ---------------------------------------------

        if torch.isnan(images).any():
            print(f"\nNaN detected in INPUT IMAGES at batch {batch_idx}")
            continue

        if torch.isnan(masks.float()).any():
            print(f"\nNaN detected in MASKS at batch {batch_idx}")
            continue

        # ---------------------------------------------
        # Forward Pass
        # ---------------------------------------------

        outputs = model(images)

        # ---------------------------------------------
        # CHECK OUTPUTS FOR NaNs
        # ---------------------------------------------

        if torch.isnan(outputs).any():
            print(f"\nNaN detected in MODEL OUTPUTS at batch {batch_idx}")
            continue

        # ---------------------------------------------
        # Compute Loss
        # ---------------------------------------------

        loss = criterion(outputs, masks)

        # ---------------------------------------------
        # CHECK LOSS FOR NaN
        # ---------------------------------------------

        if torch.isnan(loss):
            print(f"\nNaN LOSS detected at batch {batch_idx}")
            continue

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ---------------------------------------------
        # Gradient Clipping
        # ---------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_train_loss = train_loss / len(train_loader)

    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            if torch.isnan(loss):
                continue

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # =================================================
    # PRINT METRICS
    # =================================================

    print(f"\nEpoch {epoch + 1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss  : {avg_val_loss:.4f}")

    # =================================================
    # SAVE BEST MODEL
    # =================================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved!")

print("\nTraining Complete!")