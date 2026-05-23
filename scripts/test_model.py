import torch

from models.unet_model import build_unet

# CREATE MODEL

model = build_unet()

# DUMMY INPUT

# Terrain-aware HydraNet expects:
# 3 channels:
# [SAR + pseudo-elevation + pseudo-slope]

x = torch.randn(
    4,
    3,
    256,
    256
)

# FORWARD PASS

with torch.no_grad():

    y = model(x)

# PRINT SHAPES

print("Input Shape :", x.shape)

print("Output Shape:", y.shape)