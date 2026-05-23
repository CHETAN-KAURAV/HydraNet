import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        preds = torch.sigmoid(preds)

        preds = preds.squeeze(1)

        valid_mask = (targets != 255)

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        targets = targets.float()

        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()

        dice = (
                       2.0 * intersection + self.smooth
               ) / (
                       preds.sum()
                       + targets.sum()
                       + self.smooth
               )

        return 1 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, preds, targets):

        valid_mask = (targets != 255)

        preds_bce = preds.squeeze(1)[valid_mask]

        targets_bce = targets[valid_mask].float()

        bce_loss = self.bce(
            preds_bce,
            targets_bce
        )

        dice_loss = self.dice(
            preds,
            targets
        )

        total_loss = bce_loss + dice_loss

        return total_loss