import argparse
import logging
import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from evaluate import evaluate
import unet
from utils.data_loading import BasicDataset, load_image
from utils.dice_score import dice_loss

dir_img = Path('./data/HEqDRIVE/train/images/')
dir_truth = Path('./data/HEqDRIVE/train/manual/')
dir_attention_area = Path('data/HEqDRIVE/mask.gif')
dir_checkpoint = Path('./checkpoints/')


def train_model(
    model,
    device,
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    val_percent: float = 0.25,
    save_checkpoint: bool = False,
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
    train_transform = A.Compose([
        A.RandomCrop(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(256, 256),
        ToTensorV2()
    ])
    val_transform = A.Compose([A.Resize(256, 256), ToTensorV2()])
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    # 3. Create data loaders
    attention_area = load_image(dir_attention_area, True)
    loader_args = dict(batch_size=batch_size,
                       num_workers=os.cpu_count(),
                       pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set,
                            shuffle=False,
                            drop_last=False,
                            **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='drive', resume='allow', anonymous='must')
    # wandb.api.api_key = 'b3518f13f1b3184b76d233e2f2b1f7cbef587a1f'
    experiment.config.update(
        dict(epochs=epochs,
             batch_size=batch_size,
             learning_rate=learning_rate,
             val_percent=val_percent,
             save_checkpoint=save_checkpoint,
             amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5)  # goal: maximize Dice score of val set
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    val_score = 0.0
    best_val_score = 0.0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}',
                  unit='img') as pbar:  # progress bar
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

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
                    # if model.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)),
                                      true_masks.float(),
                                      multiclass=False)
                    # else:
                    #     loss = criterion(masks_pred, true_masks)
                    # loss += dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                   F.one_hot(true_masks,
                    #                             model.n_classes).permute(
                    #                                 0, 3, 1, 2).float(),
                    #                   multiclass=True)

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

                # Evaluation round
                # division_step = (n_train // (5 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not torch.isinf(value).any():
                histograms['Weights/' + tag] = wandb.Histogram(
                    value.data.cpu())
            if not torch.isinf(value.grad).any():
                histograms['Gradients/' + tag] = wandb.Histogram(
                    value.grad.data.cpu())

        val_score = evaluate(model, val_loader, device, amp)
        # scheduler.step(loss)
        scheduler.step(val_score)

        logging.info('Validation Dice score: {}'.format(val_score))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'images': wandb.Image(images[0].cpu()),
                'masks': {
                    'true': wandb.Image(true_masks[0].float().cpu()),
                    'pred':
                    wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                },
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        except:
            pass
        # save the best model
        if (epoch > epochs // 2
                and best_val_score < val_score) or save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            if best_val_score < val_score:
                best_val_score = val_score
                torch.save(
                    state_dict,
                    str(dir_checkpoint /
                        'checkpoint_epoch{}_best.pth'.format(epoch)))
                logging.info(f'New best model and checkpoint {epoch} saved!')
            else:
                torch.save(
                    state_dict,
                    str(dir_checkpoint /
                        'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    # save last model
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    state_dict['mask_values'] = dataset.mask_values
    torch.save(
        state_dict,
        str(dir_checkpoint / 'checkpoint_epoch{}_last.pth'.format(epoch)))
    logging.info(f'Last model and checkpoint {epoch} saved!')


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
    parser.add_argument(
        '--learning-rate',
        '-l',
        metavar='LR',
        type=float,
        default=1e-6,  #1e-5
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
                        default=True,
                        help='Use mixed precision')
    parser.add_argument('--classes',
                        '-c',
                        type=int,
                        default=1,
                        help='Number of classes')

    return parser.parse_args()


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
    model = unet.UNet()
    #load pre-trained model or existing model
    if not args.load:
        state_dict = torch.load('models/unet_pretrained_in_mri.pt',
                                    map_location=device)
        logging.info(f'Model loaded from pretrained model')
    else:
        model = model.to(memory_format=torch.channels_last)
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        logging.info(f'Model loaded from {args.load}')
    model.load_state_dict(state_dict)
    model.encoder1.enc1conv1.in_channels = 1
    model.n_channels = 1

    model.to(device=device)
    train_model(model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                val_percent=args.val / 100,
                amp=args.amp)
