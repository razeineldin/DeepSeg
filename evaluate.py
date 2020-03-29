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

# Evaluate the entire predictions
def get_truth_images(truth_dir='truth/', truth_shape=(3445,224,224)):
    truth_imgs = np.zeros(truth_shape)
    for i , img in enumerate(tqdm(glob.glob(os.path.join(truth_dir,"*.png")))):
        image = imread(img, 0)
        truth_imgs[i,] = resize(image, (224, 224), interpolation = INTER_NEAREST)
    return truth_imgs

def get_prediction_images(pred_dir='preds/', preds_shape=(3445,240,240)):
    pred_imgs = np.zeros(preds_shape)
    for i , img in enumerate(tqdm(glob.glob(os.path.join(pred_dir,"*.png")))):
        pred_imgs[i,] = imread(img, 0)
    return pred_imgs

def main(sample_output=False):
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

    # get evaluations of all images in the prediction directory
    predictions_shape = (config['n_valid_images'], config['input_height'], config['input_width'])
    predictions = np.zeros(predictions_shape)
    predictions = get_prediction_images(pred_dir=config['pred_path'], preds_shape=predictions.shape)
    truth_images = np.zeros(predictions_shape)
    truth_images = get_truth_images(truth_dir=config['val_annotations'], truth_shape=truth_images.shape)

    truth_whole = get_whole_tumor_mask(truth_images)
    #truth_core = get_tumor_core_mask(truth_images)
    #truth_enhancing = get_enhancing_tumor_mask(truth_images)
    pred_whole = get_whole_tumor_mask(predictions)
    #pred_core = get_tumor_core_mask(predictions)
    #pred_enhancing = get_enhancing_tumor_mask(predictions)

    whole_dice_score = get_dice_coefficient(truth_whole, pred_whole)
    #core_dice_score = get_dice_coefficient(truth_core, pred_core)
    #enhancing_dice_score = get_dice_coefficient(truth_enhancing, pred_enhancing)
    whole_hausdorff = get_hausdorff_distance(truth_whole, pred_whole)
    #core_hausdorff = get_hausdorff_distance(truth_core, pred_core)
    #enhancing_hausdorff = get_hausdorff_distance(truth_enhancing, pred_enhancing)
    whole_sensitivity = get_sensitivity(truth_whole, pred_whole)
    #core_sensitivity = get_sensitivity(truth_core, pred_core)
    #enhancing_sensitivity = get_sensitivity(truth_enhancing, pred_enhancing)
    whole_specificity = get_specificity(truth_whole, pred_whole)
    #core_specificity = get_specificity(truth_core, pred_core)
    #enhancing_specificity = get_specificity(truth_enhancing, pred_enhancing)

    print('Whole dice score:', whole_dice_score)
    #print('Core dice score:', core_dice_score)
    #print('Enhancing dice score:', enhancing_dice_score)
    print('Whole hausdorff distance (mm):', whole_hausdorff)
    #print('Core hausdorff distance (mm):', core_hausdorff)
    #print('Enhancing hausdorff distance (mm):', enhancing_hausdorff)
    print('Whole sensitivity:', whole_sensitivity)
    #print('Core sensitivity:', core_sensitivity)
    #print('Enhancing sensitivity:', enhancing_sensitivity)
    print('Whole specificity:', whole_specificity)
    #print('Core specificity:', core_specificity)
    #print('Enhancing specificity:', enhancing_specificity)

    # sample output
    if sample_output:
        sample_path = config['sample_path']
        orig_path = config['val_images']+config['train_modality'][0]+sample_path +'.png' # T1 image
        truth_path = config['val_annotations']+sample_path+'.png'
        pred_path = "out_test_file/"+sample_path+"_pred.png"
        pred_img = predict(unet_2d_model, inp = orig_path, out_fname="out_test_file/"+sample_path+"_pred.png")

        # load as grayscale images
        orig_img = imread(orig_path, 0)
        truth_img = imread(truth_path, 0)
        pred_img = imread(pred_path, 0)

        unique, counts = np.unique(truth_img, return_counts=True)
        print('Truth', dict(zip(unique, counts)))
        unique, counts = np.unique(pred_img, return_counts=True)
        print('Preds', dict(zip(unique, counts)))

        f = plt.figure()
        # (nrows, ncols, index)
        f.add_subplot(1,3, 1)
        plt.title('Original image')
        plt.imshow(orig_img, cmap='gray')
        f.add_subplot(1,3, 2)
        plt.title('Predicted image')
        plt.imshow(pred_img)
        f.add_subplot(1,3, 3)
        plt.title('Ground truth image')
        plt.imshow(truth_img)
        plt.show(block=True)

        truth_img = resize(truth_img, (224, 224), interpolation = INTER_NEAREST)
        truth_whole = get_whole_tumor_mask(truth_img)
        #truth_core = get_tumor_core_mask(truth_img)
        #truth_enhancing = get_enhancing_tumor_mask(truth_img)
        pred_img = resize(pred_img, (224, 224), interpolation=INTER_NEAREST)
        pred_whole = get_whole_tumor_mask(pred_img)
        #pred_core = get_tumor_core_mask(pred_img)
        #pred_enhancing = get_enhancing_tumor_mask(pred_img)

        whole_dice_score = get_dice_coefficient(truth_whole, pred_whole)
        #core_dice_score = get_dice_coefficient(truth_core, pred_core)
        #enhancing_dice_score = get_dice_coefficient(truth_enhancing, pred_enhancing)
        whole_hausdorff = get_hausdorff_distance(truth_whole, pred_whole)
        #core_hausdorff = get_hausdorff_distance(truth_core, pred_core)
        #enhancing_hausdorff = get_hausdorff_distance(truth_enhancing, pred_enhancing)
        whole_sensitivity = get_sensitivity(truth_whole, pred_whole)
        #core_sensitivity = get_sensitivity(truth_core, pred_core)
        #enhancing_sensitivity = get_sensitivity(truth_enhancing, pred_enhancing)
        whole_specificity = get_specificity(truth_whole, pred_whole)
        #core_specificity = get_specificity(truth_core, pred_core)
        #enhancing_specificity = get_specificity(truth_enhancing, pred_enhancing)

        print('Whole dice score:', whole_dice_score)
        #print('Core dice score:', core_dice_score)
        #print('Enhancing dice score:', enhancing_dice_score)
        print('Whole hausdorff distance (mm):', whole_hausdorff)
        #print('Core hausdorff distance (mm):', core_hausdorff)
        #print('Enhancing hausdorff distance (mm):', enhancing_hausdorff)
        print('Whole sensitivity:', whole_sensitivity)
        #print('Core sensitivity:', core_sensitivity)
        #print('Enhancing sensitivity:', enhancing_sensitivity)
        print('Whole specificity:', whole_specificity)
        #print('Core specificity:', core_specificity)
        #print('Enhancing specificity:', enhancing_specificity)

if __name__ == "__main__":
    main(sample_output=config['sample_output'])
