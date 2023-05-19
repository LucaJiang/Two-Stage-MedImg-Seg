import numpy as np
import cv2, os, shutil
from PIL import Image

def preprocess(img, mask, clahe):
    '''
    img: PIL image
    mask: PIL image
    clahe: cv2 clahe object
    return: PIL image
    '''
    r, g, b = img.convert("RGB").split()
    new_img = clahe.apply(np.asarray(g))
    masked_img = Image.fromarray(new_img*(np.asarray(mask)==255), "L")
    return masked_img

shutil.copytree('data/DRIVE', 'data/HEqDrive')

# define clahe and read mask
clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(2, 2))
mask = Image.open('data/DRIVE/mask.gif').convert("L")

# process train data
for file in os.listdir('data/HEqDrive/train/images'):
    img = Image.open('data/HEqDrive/train/images/'+file)
    img = preprocess(img, mask, clahe)
    img.save('data/HEqDrive/train/images/'+file)

# process test data
for file in os.listdir('data/HEqDrive/test/images'):
    img = Image.open('data/HEqDrive/test/images/'+file)
    img = preprocess(img, mask, clahe)
    img.save('data/HEqDrive/test/images/'+file)

print('Preprocess done!')
