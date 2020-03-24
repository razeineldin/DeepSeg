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
import six
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import imread, imwrite, resize, INTER_NEAREST
from keras import backend as K

K.set_image_data_format('channels_last')
if K.image_data_format() == 'channels_first':
    IMAGE_ORDERING = 'channels_first'
elif K.image_data_format() == 'channels_last':
    IMAGE_ORDERING = 'channels_last'

def predict(model=None, inp=None, out_fname=None):
    output_width = model.output_width
    output_height  = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
 
    if(len(config['train_modality'])==1):
        arr = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING) # (224, 224, 3)
    else:
        arr = get_images_arr(inp, input_width, input_height, odering=IMAGE_ORDERING) # (224, 224, n_modalities)
    
    pr = model.predict(np.array([arr]))[0] # (50176, 2)
    # comapare the two channels and get the max value (with 1 in new array)
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2) # (224, 224)

    # change the predicted label 3 back to value of 4 (standard BraTS labels)
    pr[pr==3] = 4

    if not out_fname is None:
        imwrite(out_fname, pr)
    return pr

def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None, checkpoints_path=None, train_modalities=config['train_modality']):
    if inps is None and (not inp_dir is None):
        inps = glob.glob(os.path.join(inp_dir,"*.png"))

    assert type(inps) is list
    all_prs = []
    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types)  :
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else :
                out_fname = os.path.join(out_dir, str(i)+ ".jpg")

        pr = predict(model, inp,out_fname)
        all_prs.append(pr)
    return all_prs

# get predictions of all images in the validation directory
unet_2d_model = get_deepseg_model(
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

predict_multiple(
    unet_2d_model,
    inp_dir = config['val_images']+config['train_modality'][0], 
    out_dir = config['pred_path'],
    train_modalities = config['train_modality']
)

# sample output
### BRATS 2019
sample_path1 = 'BraTS19_TCIA09_462_1-70' # LGG
sample_path = 'BraTS19_TCIA10_408_1-50' # HGG

orig_path1 = config['val_images']+config['train_modality'][0]+ sample_path1 +'.png' # FLAIR image
orig_path = config['val_images']+config['train_modality'][0]+ sample_path +'.png' # FLAIR image

truth_path = config['val_annotations']+ sample_path+'.png'
truth_path1 = config['val_annotations']+ sample_path1+'.png'

pred_img = predict(unet_2d_model, inp= orig_path)
pred_img1 = predict(unet_2d_model, inp= orig_path1)

print(pred_img.shape)

# load as grayscale images
orig_img = imread(orig_path, 0)
orig_img1 = imread(orig_path1, 0)

truth_img = imread(truth_path, 0)
truth_img = resize(truth_img, (224, 224), interpolation = INTER_NEAREST)

truth_img1 = imread(truth_path1, 0)
truth_img1 = resize(truth_img1, (224, 224), interpolation = INTER_NEAREST)

f = plt.figure()
# (nrows, ncols, index)
f.add_subplot(2,3, 1)
plt.title('Original image')
plt.imshow(orig_img, cmap='gray')
f.add_subplot(2,3, 2)
plt.title('Predicted image')
plt.imshow(pred_img)
f.add_subplot(2,3, 3)
plt.title('Truth image')
plt.imshow(truth_img)

f.add_subplot(2,3, 4)
plt.title('Original image')
plt.imshow(orig_img1, cmap='gray')
f.add_subplot(2,3, 5)
plt.title('Predicted image')
plt.imshow(pred_img1)
f.add_subplot(2,3, 6)
plt.title('Truth image')
plt.imshow(truth_img1)
plt.show(block=True)
