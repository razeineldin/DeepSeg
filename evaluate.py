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
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

def save_evaluation_csv(pred_path='preds/', truth_path='truth/', evaluate_path='evaluations/', save_plot=False):
    header = ("Dice", "Sensitivity", "Specificity", "Hausdorff distance")
    evaluation_functions = (get_dice_coefficient, get_sensitivity, get_specificity, get_hausdorff_distance)
    rows = list()
    subject_ids = list()

    for i, img in enumerate(tqdm(glob.glob(pred_path+"/*"))):
        subject_ids.append(os.path.basename(img))
        prediction = imread(img, 0)
        truth_image = imread(truth_path+os.path.basename(img), 0)
        truth_image = resize(truth_image, (config['input_height'], config['input_width']), interpolation = INTER_NEAREST)

        truth_whole = get_whole_tumor_mask(truth_image)
        #truth_core = get_tumor_core_mask(truth_image)
        #truth_enhancing = get_enhancing_tumor_mask(truth_image)
        pred_whole = get_whole_tumor_mask(prediction)
        #pred_core = get_tumor_core_mask(prediction)
        #pred_enhancing = get_enhancing_tumor_mask(prediction)

        rows.append([func(truth_whole, pred_whole)for func in evaluation_functions])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv(evaluate_path+"/brats19_"+config['project_name']+"_scores.csv")

    if save_plot:
        scores = dict()
        for index, score in enumerate(df.columns[:-1]):
            values = df.values.T[index]
            scores[score] = values[np.isnan(values) == False]

        plt.boxplot(list(scores.values()), labels=list(scores.keys()))
        plt.ylabel("Evaluation scores")
        plt.savefig(evaluate_path+"/brats19_"+config['project_name']+"_scores_boxplot.png")
        plt.close()

def main(evaluate_val=True, evaluate_val_nifti=True, evaluate_keras=False, save_csv=False, sample_output=False):
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

    # evaluate the entire predictions
    if evaluate_val:
        print("Evaluating the whole predictions:")
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

        evaluation_functions = (get_dice_coefficient, get_sensitivity, get_specificity, get_hausdorff_distance)
        print("Dice coefficient, Sensitivity, Specificity, Hausdorff distance")
        print([func(truth_whole, pred_whole)for func in evaluation_functions])

    # evaluate the entire predictions (nifti)
    if evaluate_val_nifti:
        header = ("Dice", "Sensitivity", "Specificity", "Hausdorff distance")
        evaluation_functions = (get_dice_coefficient, get_sensitivity, get_specificity, get_hausdorff_distance)
        rows = list()
        subject_ids = list()

        for i, case_folder in enumerate(tqdm(glob.glob(config['valid_cases_dir']+"/*"))):
            #print("case_folder", case_folder)
            case_ID = os.path.basename(case_folder)
            subject_ids.append(case_ID)
            #print("case_ID", case_ID)

            truth_file = os.path.join(case_folder, case_ID+"_truth.nii.gz")
            truth_image = nib.load(truth_file)
            truth = truth_image.get_data()
            #truth = resize(truth, (config['input_height'], config['input_width']), interpolation = INTER_NEAREST)
            prediction_file = config['pred_path_nifti_240'] +'/'+ "%s.nii.gz"%(case_ID)
            prediction_image = nib.load(prediction_file)
            prediction = prediction_image.get_data()
 
            truth_whole = get_whole_tumor_mask(truth)
            #truth_core = get_tumor_core_mask(truth)
            #truth_enhancing = get_enhancing_tumor_mask(truth)
            pred_whole = get_whole_tumor_mask(prediction)
            #pred_core = get_tumor_core_mask(prediction)
            #pred_enhancing = get_enhancing_tumor_mask(prediction)

            rows.append([func(truth_whole, pred_whole)for func in evaluation_functions])
            #print([func(truth_whole, pred_whole)for func in evaluation_functions])

        df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        df.to_csv(config['evaluate_path']+"/brats19_"+config['project_name']+"_scores_nifti.csv")

        if config['save_plot']:
            scores = dict()
            for index, score in enumerate(df.columns[:-1]):
                values = df.values.T[index]
                scores[score] = values[np.isnan(values) == False]

            plt.boxplot(list(scores.values()), labels=list(scores.keys()))
            plt.ylabel("Evaluation scores")
            plt.savefig(config['evaluate_path']+"/brats19_"+config['project_name']+"_scores_nifti_boxplot.png")
            plt.close()

    # evaluate using keras 
    if evaluate_keras:
        val_generator = image_segmentation_generator(config['val_images'], config['val_annotations'],  config['val_batch_size'], config['classes'], config['input_height'], config['input_width'], config['output_height'], config['output_width'], do_augment=False, shuffle=False)

        results = unet_2d_model.evaluate_generator(val_generator, steps=config['validation_steps'], verbose=1, max_queue_size=1, workers=1, use_multiprocessing=False) 
        print(results)

    # save data to .csv file
    if save_csv:
        print("Saving the evaluations to an .csv file:")
        save_evaluation_csv(pred_path=config['pred_path'], truth_path=config['val_annotations'],
                            evaluate_path=config['evaluate_path'], save_plot=config['save_plot'])

    # sample output
    if sample_output:
        sample_path = config['sample_path']
        print("Evaluating BraTS 19 sample:", sample_path)
        orig_path = config['val_images']+config['train_modality'][0]+sample_path +'.png' # T1 image
        truth_path = config['val_annotations']+sample_path+'.png'
        pred_path = "out_test_file/"+sample_path+"_pred.png"
        pred_img = predict(unet_2d_model, inp = orig_path, out_fname="out_test_file/"+sample_path+"_pred.png")

        # load as grayscale images
        orig_img = imread(orig_path, 0)
        truth_img = imread(truth_path, 0)
        truth_img = resize(truth_img, (config['input_height'], config['input_width']), interpolation = INTER_NEAREST)
        pred_img = imread(pred_path, 0)

        unique, counts = np.unique(truth_img, return_counts=True)
        print('Truth', dict(zip(unique, counts)))
        unique, counts = np.unique(pred_img, return_counts=True)
        print('Preds', dict(zip(unique, counts)))

        truth_whole = get_whole_tumor_mask(truth_img)
        #truth_core = get_tumor_core_mask(truth_img)
        #truth_enhancing = get_enhancing_tumor_mask(truth_img)
        pred_whole = get_whole_tumor_mask(pred_img)
        #pred_core = get_tumor_core_mask(pred_img)
        #pred_enhancing = get_enhancing_tumor_mask(pred_img)

        evaluation_functions = (get_dice_coefficient, get_hausdorff_distance, get_sensitivity, get_specificity)
        print("Whole Dice, Hausdorff distance, Sensitivity, Specificity")
        print([func(truth_whole, pred_whole)for func in evaluation_functions])

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

if __name__ == "__main__":
    main(config['evaluate_val'], config['evaluate_val_nifti'], config['evaluate_keras'], config['save_csv'], config['sample_output'])
