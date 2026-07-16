import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(1)

        dice = (
            2 * intersection + self.smooth
        ) / (
            pred.sum(1) +
            target.sum(1) +
            self.smooth
        )

        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):

        return (
            0.5 * self.bce(pred, target)
            +
            0.5 * self.dice(pred, target)
        )