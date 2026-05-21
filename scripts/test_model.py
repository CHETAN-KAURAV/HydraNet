import torch

from models.unet_model import build_unet

# -----------------------------
# Create Model
# -----------------------------
model = build_unet()

# -----------------------------
# Dummy Input
# -----------------------------
x = torch.randn(4, 1, 256, 256)

# -----------------------------
# Forward Pass
# -----------------------------
with torch.no_grad():
    y = model(x)

# -----------------------------
# Print Output Shape
# -----------------------------
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)