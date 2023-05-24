import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils.dice_score import dice_coeff
from utils.metrics import classwise_iou, classwise_f1


@torch.no_grad()
def evaluate(net, dataloader, criterion, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            loss += criterion(mask_pred.squeeze(1), mask_true.float())
    net.train()
    return loss / max(num_val_batches, 1)


@torch.no_grad()
def evaluate_all(net, dataloader, criterion, device, amp):
    '''
    return loss, dice_score, iou, f1, auc
    '''
    net.eval()
    num_batches = len(dataloader)
    dice_score = 0
    iou = 0
    f1 = 0
    loss = 0
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Eval round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.squeeze(1)
            mask_pred_threshold = (torch.sigmoid(mask_pred) > 0.5).float()
            # calculate metrics
            dice_score += dice_coeff(mask_pred_threshold, mask_true, reduce_batch_first=False)
            iou += classwise_iou(mask_pred_threshold, mask_true)
            f1 += classwise_f1(mask_pred_threshold, mask_true)
            loss += criterion(mask_pred.squeeze(1), mask_true.float())
    net.train()
    return loss / max(num_batches, 1), dice_score / max(num_batches, 1), iou / max(num_batches, 1), f1 / max(num_batches, 1)

