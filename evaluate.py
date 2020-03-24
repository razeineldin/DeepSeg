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

from config import *
from data import *
from utils import *
from models import *
from predict import *
from keras import backend as K

unet_2d_model =  get_deepseg_model(
        encoder_name=config['encoder_name'], 
        decoder_name=config['decoder_name'], 
        n_classes=config['n_classes'], 
        input_height=config['input_height'], 
        input_width=config['input_width'], 
        depth=config['model_depth'], 
        filter_size=config['filter_size'], 
        up_layer=config['up_layer'],
        trainable=config['trainable'], 
        load_model=config['load_model'])

# sample_path = 'BraTS19_TCIA03_257_1_75' # Low dice
# sample_path = 'BraTS19_TCIA03_296_1_75' # Very good dice
# sample_path = 'BraTS19_TCIA08_205_1_100'
# sample_path = 'BraTS19_TCIA10_175_1-75'
sample_path = 'BraTS19_TCIA10_408_1-66'

orig_path = config['val_images']+config['train_modality'][0]+sample_path +'.png' # T1 image

truth_path = config['val_annotations']+sample_path+'.png'
pred_path = "out_test_file/"+sample_path+"_pred.png"
pred_img = predict(unet_2d_model, inp = orig_path, out_fname="out_test_file/"+sample_path+"_pred.png")

# load as grayscale images
orig_img = imread(orig_path, 0)
truth_img = imread(truth_path, 0)
pred_img2 = imread(pred_path, 0)

unique, counts = np.unique(truth_img, return_counts=True)
print('Truth', dict(zip(unique, counts)))
unique, counts = np.unique(pred_img, return_counts=True)
print('Preds', dict(zip(unique, counts)))

f = plt.figure()
# (nrows, ncols, index)
f.add_subplot(1,4, 1)
plt.title('Original image')
plt.imshow(orig_img, cmap='gray')
f.add_subplot(1,4, 2)
plt.title('Predicted image')
plt.imshow(pred_img)
f.add_subplot(1,4, 3)
plt.title('Ground truth image')
plt.imshow(truth_img)
f.add_subplot(1,4, 4)
plt.title('Loaded predict image')
plt.imshow(pred_img2)
plt.show(block=True)

truth_img = cv2.resize(truth_img, (224, 224), interpolation = cv2.INTER_NEAREST)#, interpolation=cv2.INTER_NEAREST)
truth_whole = get_whole_tumor_mask(truth_img)
truth_core = get_tumor_core_mask(truth_img)
truth_enhancing = get_enhancing_tumor_mask(truth_img)

pred_img = cv2.resize(pred_img, (224, 224), interpolation=cv2.INTER_NEAREST)
pred_whole = get_whole_tumor_mask(pred_img)
pred_core = get_tumor_core_mask(pred_img)
pred_enhancing = get_enhancing_tumor_mask(pred_img)

whole_dice_score = evaluate_dice_coefficient(truth_whole, pred_whole)
core_dice_score = evaluate_dice_coefficient(truth_core, pred_core)
enhancing_dice_score = evaluate_dice_coefficient(truth_enhancing, pred_enhancing)

print('whole_dice_score', whole_dice_score) # 0.8480442053220881, 0.628686327077748
print('core_dice_score', core_dice_score)
print('enhancing_dice_score', enhancing_dice_score)
