import torch
from torch import Tensor, nn


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim) # both are one
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - dice_coeff(input, target, reduce_batch_first=True)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        dice_coef = (2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
        loss = 1. - dice_coef
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, input, target):
        # Binary Cross Entropy
        bce_loss = nn.BCEWithLogitsLoss(weight=None, reduction='mean')(input, target)

        # Dice Loss
        probas = torch.sigmoid(input)
        inter = torch.sum(target * probas)
        union = torch.sum(target) + torch.sum(probas) + 1e-6
        dice_loss = 1 - (2 * inter / union)

        # Weighted sum
        loss = (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss)

        return loss
