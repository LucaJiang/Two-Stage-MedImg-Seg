import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask
from unet import UNet
import albumentations as A


test_transform = A.Compose([
    A.Resize(height=256, width=256, interpolation=3),
    A.Normalize(mean=(0.5, ), std=(0.5, ))
])


def predict_img_in_file(net, file_name, device):
    net.eval()

    # img = torch.from_numpy(
    #     BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = Image.open(file_name)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # output.squeeze() #! none
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]),
        #                        mode='bilinear')
        # if net.n_classes > 1:
        #     mask = output.argmax(dim=1)
        # else:  # net.n_classes == 1
        #     mask = torch.sigmoid(output) > out_threshold  #!

    return output[0].float().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model',
                        '-m',
                        # default='./models/model.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored',
                        required=True)
    parser.add_argument('--input_image',
                        '-i',
                        metavar='INPUT',
                        nargs='+',
                        help='Filenames of input images',
                        required=True)
    parser.add_argument('--input_mask',
                        '-im',
                        metavar='INPUTMASK',
                        nargs='+',
                        help='Filenames of input masks',
                        required=True)
    parser.add_argument('--output',
                        '-o',
                        metavar='OUTPUT',
                        nargs='+',
                        help='Filenames of output images')
    # parser.add_argument('--viz',
    #                     '-v',
    #                     action='store_true',
    #                     help='Visualize the images as they are processed')
    # parser.add_argument('--no-save',
    #                     '-n',
    #                     action='store_true',
    #                     help='Do not save the output masks')
    parser.add_argument(
        '--mask-threshold',
        '-t',
        type=float,
        default=0.5,
        help='Minimum probability value to consider a mask pixel white')

    return parser.parse_args()


def get_output_filenames(args):

    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.gif'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])),
                       dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.info(f'Using model {args.model}, load from {args.input_image}, save to {args.output}')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # get data
    dataset = BasicDataset(args.input_image, args.input_mask, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # get name of images
    names = os.listdir(args.input_image)
    # get model
    net = UNet(in_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')
    # predict
    for i, data in enumerate(test_loader):
        img, mask = data
        img = img.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        mask_pred = net(img)
        # mask_pred = torch.sigmoid(mask_pred)
        mask_pred = mask_pred.squeeze().cpu().numpy()
        # mask_pred = (mask_pred > 0.5).astype(np.uint8)
        mask_pred = mask_pred * 255
        cv2.imwrite(os.path.join(args.output, f'{names[i]}_pred.png'), mask_pred)
        print(f'{i}th image done!')
    print('All done!')
