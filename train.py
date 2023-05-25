import argparse
import logging
import os, sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import albumentations as A
from tqdm import tqdm
import wandb
from evaluate import evaluate, evaluate_all
import unet
from utils.data_loading import BasicDataset, load_image
from utils.dice_score import DiceBCELoss, DiceLoss
from utils.elastic_loss import EnergyLoss
from utils.active_contour_loss import ACLoss, ACLossV2
from utils.nll_loss import LogNLLLoss
# from utils.metrics import classwise_iou, classwise_f1

dir_img = Path('./data/HEqDRIVE/train/images/')
dir_truth = Path('./data/HEqDRIVE/train/manual/')
dir_attention_area = Path('data/HEqDRIVE/mask.gif')
dir_checkpoint = Path('./checkpoints/')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--epochs',
                        '-e',
                        metavar='E',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size',
                        '-b',
                        dest='batch_size',
                        metavar='B',
                        type=int,
                        default=4,
                        help='Batch size')
    parser.add_argument('--loss',
                        '-l',
                        type=str,
                        default='bce',
                        help='Loss function')
    parser.add_argument(
        '--learning-rate',
        '-lr',
        metavar='LR',
        type=float,
        default=1e-4,  #1e-5
        help='Learning rate',
        dest='lr')
    parser.add_argument('--load',
                        '-f',
                        type=str,
                        default=False,
                        help='Load model from a .pth file')
    parser.add_argument(
        '--validation',
        '-v',
        dest='val',
        type=float,
        default=25.0,
        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='Use mixed precision')
    return parser.parse_args()


def train_model(  # do not change params here
    model,
    device,
    epochs: int = 100,
    batch_size: int = 4,
    loss_type: str = 'bce',
    learning_rate: float = 1e-4,
    val_percent: float = 0.25,
    save_checkpoint: bool = False,  # save all checkpoints
    amp: bool = True,
    weight_decay: float = 1e-8,
    momentum: float = 0.9,
    gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_truth)

    # 2. Split into train / validation partitions and set augmentation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # train_transform = transforms.Compose([
    #     transforms.RandomApply([
    #     transforms.RandomCrop(size=128, padding=4)
    #     ], p=0.5),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.Resize(256),
    #     transforms.ToTensor()
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.ToTensor()
    # ])
    resize = 256
    train_transform = A.Compose([
        A.RandomCrop(height=128, width=128, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=resize, width=resize, interpolation=3),
        A.Normalize(mean=(0.5, ), std=(0.5, )),
        # ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=resize, width=resize, interpolation=3),
        A.Normalize(mean=(0.5, ), std=(0.5, )),
        # ToTensorV2()
    ])
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    # 3. Create data loaders
    # attention_area = load_image(dir_attention_area)
    loader_args = dict(batch_size=batch_size,
                       num_workers=os.cpu_count(),
                       pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set,
                            shuffle=False,
                            drop_last=False,
                            **loader_args)

    # (Initialize logging)
    experiment = wandb.init(entity='jiangwx7',
                            project='DRIVE',
                            anonymous='never',
                            resume='allow')
    experiment.config.update(
        dict(epochs=epochs,
             batch_size=batch_size,
             loss=loss_type,
             learning_rate=learning_rate,
             val_percent=val_percent,
             save_checkpoint=save_checkpoint,
             amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Loss:            {loss_type}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)  # goal: minimize val-loss
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # set loss function
    if loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'energy':
        criterion = EnergyLoss(cuda=True, alpha=0.35, sigma=0.25)
    elif loss_type == 'ac':
        criterion = ACLoss(classes=1, device=device)
    elif loss_type == 'ac2':
        criterion = ACLossV2(classes=1, device=device)
    elif loss_type == 'nll':
        criterion = LogNLLLoss()
    elif loss_type == 'dice':
        criterion = DiceLoss()
    elif loss_type == 'dice_bce':
        criterion = DiceBCELoss()
    else:
        raise NotImplementedError('Loss {} not implemented'.format(loss_type))
    global_step = 0
    val_score = 0.0
    best_val_score = 0.0
    best_state_dict = None

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}',
                  unit='img') as pbar:  # progress bar
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                if global_step == 0:
                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                images = images.to(device=device,
                                   dtype=torch.float32,
                                   memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(
                        device.type if device.type != 'mps' else 'cpu',
                        enabled=amp):
                    masks_pred = model(images)
                    # print(masks_pred.shape, true_masks.shape)
                    # torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256])
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # histograms = {}
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('/', '.')
        #     if not torch.isinf(value).any():
        #         histograms['Weights/' + tag] = wandb.Histogram(
        #             value.data.cpu())
        #     if not torch.isinf(value.grad).any():
        #         histograms['Gradients/' + tag] = wandb.Histogram(
        #             value.grad.data.cpu())
        if epoch % 3 == 0:
            val_score = evaluate(model, val_loader, criterion, device, amp)
            # scheduler.step(loss)
            scheduler.step(val_score)

            logging.info('Validation loss score: {}'.format(val_score))
            if loss_type == 'energy':
                _masks_pred0 = masks_pred[0].float().cpu().detach().numpy()
                _masks_pred0 = _masks_pred0 / np.max(_masks_pred0) * 255.0
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation loss': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true':
                        wandb.Image(true_masks[0].float().cpu()),
                        'pred':
                        wandb.Image(masks_pred[0].float().cpu() * 255.0) if
                        loss_type != 'energy' else wandb.Image(_masks_pred0),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    # **histograms
                })
            except:
                pass
            # save the best model
            if (epoch > epochs // 2
                    and best_val_score < val_score) or save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                if best_val_score < val_score:
                    best_state_dict = model.state_dict()
                    best_val_score = val_score
                    torch.save(
                        best_state_dict,
                        str(dir_checkpoint /
                            '{}_epoch{}_best.pth'.format(loss_type, epoch)))
                    logging.info(
                        f'New best model and checkpoint {epoch} saved!')
                else:
                    torch.save(
                        model.state_dict(),
                        str(dir_checkpoint /
                            'checkpoint_epoch{}.pth'.format(epoch)))
                    logging.info(f'Checkpoint {epoch} saved!')
    # 6. save last model
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(
        state_dict,
        str(dir_checkpoint / 'checkpoint_epoch{}_last.pth'.format(epoch)))
    logging.info(f'Last model and checkpoint {epoch} saved!')
    # 7. compute metrics on train and val set
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    train_loss, train_dice_score, train_iou, train_f1 = evaluate_all(
        model, train_loader, criterion, device, amp)
    val_loss, val_dice_score, val_iou, val_f1 = evaluate_all(
        model, val_loader, criterion, device, amp)
    logging.info(
        f'Train loss: {train_loss:.4f} | Train dice score: {train_dice_score:.4f} | Train iou: {train_iou:.4f} | Train f1: {train_f1:.4f}'
    )
    logging.info(
        f'Val loss: {val_loss:.4f} | Val dice score: {val_dice_score:.4f} | Val iou: {val_iou:.4f} | Val f1: {val_f1:.4f}'
    )
    experiment.log({
        'train loss': train_loss,
        'train dice score': train_dice_score,
        'train iou': train_iou,
        'train f1': train_f1,
        'val loss': val_loss,
        'val dice score': val_dice_score,
        'val iou': val_iou,
        'val f1': val_f1,
    })
    if os.path.exists('record.csv'):
        record = pd.read_csv('record.csv', header=0)
        record = pd.concat([
            record,
            pd.Series([
                train_loss.cpu().numpy(),
                train_dice_score.cpu().numpy(),
                train_iou.cpu().numpy(),
                train_f1.cpu().numpy(),
                val_loss.cpu().numpy(),
                val_dice_score.cpu().numpy(),
                val_iou.cpu().numpy(),
                val_f1.cpu().numpy()
            ],
                      index=record.columns)
        ])
    else:
        record = pd.DataFrame(columns=[
            'train loss', 'train dice score', 'train iou', 'train f1',
            'val loss', 'val dice score', 'val iou', 'val f1'
        ])
        record = pd.concat([
            record,
            pd.Series([
                train_loss.cpu().numpy(),
                train_dice_score.cpu().numpy(),
                train_iou.cpu().numpy(),
                train_f1.cpu().numpy(),
                val_loss.cpu().numpy(),
                val_dice_score.cpu().numpy(),
                val_iou.cpu().numpy(),
                val_f1.cpu().numpy()
            ],
                      index=record.columns)
        ])
    record.to_csv('record.csv', index=False)

    experiment.finish()


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # not use

    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    loss_type = args.loss
    #load pre-trained model or existing model
    if not args.load:
        model = unet.UNet()
        state_dict = torch.load('pretrain_params/unet_pretrained_in_mri.pt',
                                map_location=device)
        logging.info(f'Model loaded from pre-trained MRI model')
        model.load_state_dict(state_dict)
        # model.encoder1.enc1conv1.in_channels = 1
        model.encoder1.enc1conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        model.n_channels = 1
    else:
        model = unet.UNet(in_channels=1)
        model = model.to(memory_format=torch.channels_last)
        state_dict = torch.load(args.load, map_location=device)
        logging.info(f'Model loaded from {args.load}.')
        model.load_state_dict(state_dict)
    model.to(device=device)

    train_model(model=model,
                epochs=args.epochs,
                loss_type=loss_type,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                val_percent=args.val / 100,
                amp=args.amp)
    print('Finished Training.')
