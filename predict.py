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
import nibabel as nib

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

def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None, checkpoints_path=None, train_modalities=config['train_modality'], overwrite=False):
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

        if not os.path.exists(out_fname):
            pr = predict(model, inp,out_fname)
            all_prs.append(pr)
        elif overwrite:
            pr = predict(model, inp,out_fname)
            all_prs.append(pr)
    return all_prs

def main(sample_output=False, predict_val=True, predict_val_nifti=False):
    # create the DeepSeg model
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

    # get predictions of all images in the validation directory
    if predict_val:
        predict_multiple(
            unet_2d_model,
            inp_dir = config['val_images']+config['train_modality'][0], 
            out_dir = config['pred_path'],
            train_modalities = config['train_modality'],
            overwrite = False
        )

    # get predictions of all images in the validation directory as nifti
    if predict_val_nifti:
        f=open(config['val_cases_file'], "r")
        valid_ids =f.read()
        f.close

        valid_dirs = valid_ids.split("\n")
        del valid_dirs[-1]
        for i, ID in enumerate(tqdm(valid_dirs)):
            ID_name = os.path.basename(ID)
            #print(i,ID_name)
            img = config['valid_cases_dir'] + ID_name +'/' + ID_name + '_flair.nii.gz'
            val_img = nib.load(img)
            val_data = val_img.get_fdata()
            #print("img: ", img)
            #print("val_img.shape: ", val_img.shape)
            pred_data = np.zeros((240, 240, 155))

            for n in range (155):
                #if n==1: break
                tmp_val_img = np.zeros((240, 240, 3))
                for ch in range(3):
                    tmp_val_img[:,:,ch] = val_data[:,:,n]  # 240 x 240
                tmp_val_img = resize(tmp_val_img, (224, 224), interpolation = INTER_NEAREST)
                tmp_val_img = tmp_val_img.reshape(1, 224, 224, 3)
 
                tmp_val_img = (tmp_val_img/tmp_val_img.max())*255 # scale to be 0 to 255 (uint8)
                tmp_val_img = tmp_val_img.astype(np.uint8)

                img_mean = tmp_val_img.mean() # normalization
                img_std = tmp_val_img.std()
                if(img_std != 0): tmp_val_img = (tmp_val_img - img_mean) / img_std
                else: tmp_val_img = (tmp_val_img - img_mean)

                pr = unet_2d_model.predict(tmp_val_img)[0] # (50176, 2)
                pr = pr.reshape((config['output_height'],  config['output_width'],
                                     config['n_classes'])).argmax(axis=2) # (224, 224)
                #print("pr.shape: ", pr.shape)
                pred_data[:,:,n] = resize(pr, (240, 240), interpolation = INTER_NEAREST)
    
            pred_img = nib.Nifti1Image(pred_data, val_img.affine, val_img.header)
            #print("pred_img.shape: ", pred_img.shape)
            nib.save(pred_img, config['pred_path_nifti_240'] +'/'+ "%s.nii.gz"%(ID_name))

    # sample output
    if sample_output:
        # BRATS 2019
        sample_lgg_path = 'BraTS19_TCIA09_462_1-70' # LGG
        sample_hgg_path = 'BraTS19_TCIA10_408_1-50' # HGG
        orig_lgg_path = config['val_images']+config['train_modality'][0]+ sample_lgg_path +'.png' # FLAIR image
        orig_hgg_path = config['val_images']+config['train_modality'][0]+ sample_hgg_path +'.png' # FLAIR image
        truth_lgg_path = config['val_annotations']+ sample_lgg_path+'.png'
        truth_hgg_path = config['val_annotations']+ sample_hgg_path+'.png'
        pred_lgg_img = predict(unet_2d_model, inp= orig_lgg_path)
        pred_hgg_img = predict(unet_2d_model, inp= orig_hgg_path)

        # load as grayscale images
        orig_hgg_img = imread(orig_hgg_path, 0)
        orig_lgg_img = imread(orig_lgg_path, 0)
        truth_hgg_img = imread(truth_hgg_path, 0)
        truth_hgg_img = resize(truth_hgg_img, (224, 224), INTER_NEAREST)
        truth_lgg_img = imread(truth_lgg_path, 0)
        truth_lgg_img = resize(truth_lgg_img, (224, 224), INTER_NEAREST)

        f = plt.figure()
        # (nrows, ncols, index)
        f.add_subplot(2,3, 1)
        plt.title('Original HGG image')
        plt.imshow(orig_hgg_img, cmap='gray')
        f.add_subplot(2,3, 2)
        plt.title('Predicted HGG image')
        plt.imshow(pred_hgg_img)
        f.add_subplot(2,3, 3)
        plt.title('Truth HGG image')
        plt.imshow(truth_hgg_img)

        f.add_subplot(2,3, 4)
        plt.title('Original LGG image')
        plt.imshow(orig_lgg_img, cmap='gray')
        f.add_subplot(2,3, 5)
        plt.title('Predicted LGG image')
        plt.imshow(pred_lgg_img)
        f.add_subplot(2,3, 6)
        plt.title('Truth LGG image')
        plt.imshow(truth_lgg_img)
        plt.show(block=True)

if __name__ == "__main__":
    main(config['sample_output'], config['predict_val'], config['predict_val_nifti'])
