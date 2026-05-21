import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        # Apply sigmoid
        preds = torch.sigmoid(preds)

        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Remove ignore pixels (255)
        valid_mask = targets != 255

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        intersection = (preds * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            reduction='mean'
        )

        self.dice = DiceLoss()

    def forward(self, preds, targets):

        # BCE expects float targets
        targets_float = targets.float()

        # Ignore mask for BCE
        valid_mask = targets != 255

        preds_valid = preds.squeeze(1)[valid_mask]
        targets_valid = targets_float[valid_mask]

        bce_loss = self.bce(
            preds_valid,
            targets_valid
        )

        dice_loss = self.dice(
            preds,
            targets
        )

        total_loss = bce_loss + dice_loss

        return total_loss