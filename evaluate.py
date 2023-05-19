import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.no_grad()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    loss = 0
    criterion = nn.CrossEntropyLoss(
    ) if net.n_classes > 1 else nn.BCEWithLogitsLoss()

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            #         # Dice
            #         if net.n_classes == 1:
            #             assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            #             mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()

            #             # compute the Dice score
            #             dice_score += dice_coeff(torch.squeeze(mask_pred), mask_true, reduce_batch_first=False)
            #         else:
            #             assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
            #             # convert to one-hot format
            #             mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            #             mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #             # compute the Dice score, ignoring background
            #             dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            # net.train()
            # return dice_score / max(num_val_batches, 1)

            # CE loss
            if net.n_classes == 1:
                loss += criterion(mask_pred.squeeze(1),
                                    mask_true.float())
            else:
                loss += criterion(mask_pred, mask_true)
    net.train()
    return loss / max(num_val_batches, 1)
