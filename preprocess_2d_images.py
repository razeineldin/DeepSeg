# Copyright (c) 2019 Ramy Zeineldin
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The following code is based on the following module:
# https://github.com/ellisdg/3DUnetCNN/blob/master/brats/preprocess.py under MIT License

from config import *

import cv2
import os
import pdb
import nibabel as nib
from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer

image_modality = 'image_FLAIR'

# Training
IMG_ROOT = config['preprocessed_brats']+config['train_modality']
IMG_OUTPUT_ROOT = config['preprocessed_brats_imgs']+config['train_modality']

# Validation
#IMG_ROOT = config['preprocessed_brats_val']+config['train_modality']
#IMG_OUTPUT_ROOT = config['preprocessed_brats_val_imgs']+config['train_modality']

LABEL_ROOT = config['preprocessed_brats']+'truth'
LABEL_OUTPUT_ROOT = config['preprocessed_brats']+config['label_type']


L0 = 0		# Background
L1 = 1		# Necrotic and Non-enhancing Tumor
L2 = 2		# Edema
L3 = 4		# Enhancing Tumor

L_com = 1	# Complete Tumor	(1+2+3)
L_cor = 1	# Core Tumor	(1+3)
L_enh = 1	# Enhancing Tumor	(3)

def nii2jpg_img(img_path, output_root):
    #img_name = (img_path.split('/')[-1]).split('.')[0]) # name with flair
    img_name = ((img_path.split('/')[-1]).split('.')[0]).rsplit('_', 1)[0] # name without flair

    output_path = os.path.join(output_root, img_name)
    try:
        os.mkdir(output_root)
    except:
        pass
    try:
        os.mkdir(output_path)
    except:
        pass
    img = nib.load(img_path)
    #print("Shape of image is", img.get_fdata().shape) # shape of (240, 240, 155)

    # Obtain the dat for T1
    #img = (img.get_fdata())[:,:,:,1]
    img = (img.get_fdata())[:]
    img = (img/img.max())*255 # scale to be 0 to 255 (uint8)
    img = img.astype(np.uint8)

    for i in range(img.shape[2]):
        filename = os.path.join(output_path, img_name+'-'+str(i)+'.png')
        gray_img = img[:,:,i]
        cv2.imwrite(filename, gray_img)


def nii2jpg_label(img_path, output_root):
    #img_name = (img_path.split('/')[-1]).split('.')[0]) # name with flair
    img_name = ((img_path.split('/')[-1]).split('.')[0]).rsplit('_', 1)[0] # name without flair

    output_path = os.path.join(output_root, img_name)

    try:
        os.mkdir(output_root)
    except:
        pass
    try:
        os.mkdir(output_path)
    except:
        pass
    img = nib.load(img_path)
    img = (img.get_fdata())[:,:,:]

    img = img.astype(np.uint8)


    if(config['label_type'] == "complete"):
        # Complete tumor label (8) = label 1 + 2 + 4
        img[img > 0] = L_com
    elif(config['label_type'] == "core"):
        img[img == L1] = L_cor
        img[img == L2] = L0	# Background
        img[img == L3] = L_cor
    elif(config['label_type'] == "enhancing"):
        img[img == L1] = L0	# Background
        img[img == L2] = L0	# Background
        img[img == L3] = L_enh
    elif(config['label_type'] == "l1"):
        #img[img == L1] = L1
        img[img == L2] = L0
        img[img == L3] = L0
    elif(config['label_type'] == "l2"):
        img[img == L1] = L0
        #img[img == L2] = L2
        img[img == L3] = L0
    elif(config['label_type'] == "l3"):
        img[img == L1] = L0
        img[img == L2] = L0
        #img[img == L3] = L3

    for i in range(img.shape[2]):
        filename = os.path.join(output_path, img_name+'_'+str(i)+'.png')
        gray_img = img[:,:,i]

        cv2.imwrite(filename, gray_img)

for path in tqdm(os.listdir(IMG_ROOT)):
    #print(path)
    if path[0] == '.':
        continue
    nii2jpg_img(os.path.join(IMG_ROOT,path), IMG_OUTPUT_ROOT)
"""
for path in tqdm(os.listdir(LABEL_ROOT)):
    #print(path)
    if path[0] == '.':
        continue
    nii2jpg_label(os.path.join(LABEL_ROOT,path), LABEL_OUTPUT_ROOT)
"""
