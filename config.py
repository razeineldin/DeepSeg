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

import numpy as np
import glob, os
import itertools
import random
from keras.optimizers import Adam
from keras import backend as K

K.set_image_data_format('channels_last')
if K.image_data_format() == 'channels_first':
    IMAGE_ORDERING = 'channels_first'
elif K.image_data_format() == 'channels_last':
    IMAGE_ORDERING = 'channels_last'

config = dict()
config['dataset_path'] = '/home/ramy/Documents/Datasets/BRATS_2019/dataset_19_no_empty/'

# model configuration
config['encoder_name'] = 'VGG16' # name of the encoder: UNet, UNet-Mod, VGG16, ResNet50, MobileNet, MobileNetV2, Xception, NASNetMobile, DenseNet121
config['decoder_name'] = 'UNet-Mod' # name of the decoder: UNet, UNet-Mod
config['project_name'] = config['encoder_name'] + '_' + config['decoder_name']

config['all_modalities'] = ["image_t1/", "image_t1ce/", "image_FLAIR/", "image_t2/"]
config['train_modality'] = ["image_FLAIR/"]
config['n_modalities'] = len(config['train_modality'])
config['label_type'] = '_complete/' # _complete, _core, _enhancing, _l1, _l2, _l3
config['train_label'] = 'truth' + config['label_type']

config['classes'] = [0,1] # 0 for the background, 1 for the tumor
config['n_classes'] = len(config['classes'])
# default value for one modality is 3, otherwise equals the number of modalities
config['model_depth'] = 3 if config['n_modalities']==1 else config['n_modalities']
#config['up_layer'] = True # for models VGG16, ResNet50, MobileNet, MobileNetV2, Xception, NASNetMobile, DenseNet121
config['up_layer'] = False if config['encoder_name']=="UNet" or config['encoder_name']=="UNet-Mod" or config['encoder_name']=="VGG16" else True

# paths
config['verify_dataset'] = False
config['validate'] = True # use the validation set
config['train_images'] = config['dataset_path'] + 'train_images/'
config['train_annotations'] = config['dataset_path'] + 'train_segmentation/' + config['train_label'] 
config['val_images'] = config['dataset_path'] + 'val_images/'
config['val_annotations'] = config['dataset_path'] + 'val_segmentation/' + config['train_label'] 
config['weight_dir'] = 'weights/'
config['log_dir'] = 'logs'
config['pred_path'] = 'preds/' + config['project_name'] + '/'
config['model_checkpoints'] = os.path.join(config['weight_dir'] + config['project_name'], config['project_name'])
config['tensorboard_path'] = 'logs_tensor_board/' + config['project_name']

####################################################################
### Hyper parameter: ###
config['batch_size'] = 16 
config['val_batch_size'] = 16
config['filter_size'] = 32 # number of basic filters
config['optimizer_lr'] = 1e-4
config['optimizer_name'] = Adam(config['optimizer_lr'])
config['weights_arr'] = np.array([0.05, 1.0]) # 2 Classes
####################################################################

# training parameters
config['input_height'] = 224 # 240, 256
config['input_width'] = 224
config['output_height'] = 224
config['output_width'] = 224
config['epochs'] = 35	# number of training epochs
config['load_model'] = False # continue training from a saved checkpoint
config['model_num'] = '20' # number of the model to be loaded
config['initial_epoch'] = config['model_num'] if config['load_model'] else 0  
config['trainable'] = True # make the top layers of the model trainable or not (for transfer learning)

config['n_train_images'] = len(glob.glob(config['train_images'] + 'image_FLAIR/*')) # 13779
config['n_valid_images'] = len(glob.glob(config['val_images'] + 'image_FLAIR/*')) # 3445
config['steps_per_epoch'] = config['n_train_images'] // config['batch_size'] # 512 for fast testing
config['validation_steps'] = config['n_valid_images'] // config['val_batch_size'] # 200 for fast testing

# data augmentation parameters
config['do_augment'] = True
config['flip_H'] = 0.5
config['flip_V'] = 0.5
config['scale_X'] = (0.8, 1.2)
config['scale_Y'] = (0.8, 1.2)
config['translate_X'] = (-0.2, 0.2)
config['translate_Y'] = (-0.2, 0.2)
config['rotate'] = (-25, 25)
config['shear'] = (-8, 8)
config['elastic'] = (720, 24) # alpha=720, sigma=24
config['random_order'] = True # apply augmenters in random order

# create folders
if not os.path.exists(config['log_dir']):
    os.mkdir(config['log_dir'])
if not os.path.exists(config['weight_dir'] + config['project_name']):
    os.makedirs(config['weight_dir'] + config['project_name'])
if not os.path.exists(config['pred_path']):
    os.makedirs(config['pred_path'])
if not os.path.exists(config['tensorboard_path']):
    os.makedirs(config['tensorboard_path'])

# print configs
print("\n\n####################################################################")
print("Please cite the following paper when using DeepSeg :")
print("Zeineldin, Ramy Ashraf, et al. \"DeepSeg: Deep Neural Network Framework for Automatic Brain Tumor Segmentation using Magnetic Resonance FLAIR Images.\" arXiv preprint arXiv:XXXX.XXXXX (2020).\n\n")

print("Project name is:", config['project_name'])
print("Dataset path:", config['dataset_path'])
print("Encoder name:", config['encoder_name'])
print("Decoder name:", config['decoder_name'])
print("Training modalities:", config['train_modality'])
print("Training classes:", config['classes'])
print("Training batch size:", config['batch_size'])
print("Validation batch size:", config['val_batch_size'])
print("####################################################################\n\n")

