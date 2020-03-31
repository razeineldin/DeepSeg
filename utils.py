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
from keras import backend as K
import numpy as np
#import scipy.spatial
from scipy.spatial.distance import directed_hausdorff

# metrics and losses
def weighted_categorical_crossentropy(y_true, y_pred):
    weights = K.variable(config['weights_arr'])
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)
    # return 1.-dice_coefficient(y_true, y_pred)

def dice_argmax(y_true, y_pred, smooth=1.):    
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32") # (?, ?)
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32") # (?, 50176)
    y_true_f = K.flatten(y_true) # (?,)
    y_pred_f = K.flatten(y_pred) # (?,)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_argmax_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def sensitivity(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32") # (?, ?)
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32") # (?, 50176)
    y_true_f = K.flatten(y_true) # (?,)
    y_pred_f = K.flatten(y_pred) # (?,)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32") # (?, ?)
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32") # (?, 50176)
    y_true_f = K.flatten(y_true) # (?,)
    y_pred_f = K.flatten(y_pred) # (?,)
    true_negatives = K.sum(K.round(K.clip((1-y_true_f) * (1-y_pred_f), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true_f, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def dice_argmax_whole(y_true, y_pred, smooth=1.):    
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32") # (?, ?)
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32") # (?, 50176)
    y_true_f = K.flatten(y_true) # (?,)
    y_pred_f = K.flatten(y_pred) # (?,)

    y_true_z = K.zeros_like(y_true_f)
    y_pred_z = K.zeros_like(y_pred_f)

    y_true_whole = K.cast(K.not_equal(y_true_f, y_true_z), "float32")
    y_pred_whole = K.cast(K.not_equal(y_pred_f, y_pred_z), "float32")

    intersection = K.sum(y_true_whole * y_pred_whole)
    return (2. * intersection + smooth) / (K.sum(y_true_whole) + K.sum(y_pred_whole) + smooth)

def hausdorff_distance(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32") # (?, ?)
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32") # (?, 50176)
    y_true_f = K.flatten(y_true) # (?,)
    y_pred_f = K.flatten(y_pred) # (?,)

    hd, _, _ = directed_hausdorff(y_true_f, y_pred_f)
    return hd

# evaluation functions
def get_whole_tumor_mask(data):
    return data > 0

def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)

def get_enhancing_tumor_mask(data):
    return data == 4

def get_dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def evaluate_dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def get_sensitivity(y_true, y_pred):
    true_positives = np.sum(np.multiply(y_true, y_pred))
    possible_positives = np.sum(y_true)
    if possible_positives==0: return 1
    else: return true_positives / possible_positives
    #return true_positives / possible_positives

def get_specificity(y_true, y_pred):
    true_negatives = np.sum(np.multiply(y_true==0, y_pred==0))
    possible_negatives = np.sum(y_true==0)
    if possible_negatives==0: return 1
    else: return true_negatives / possible_negatives
    #return true_negatives / possible_negatives

def get_hausdorff_distance(truth, prediction):
    """Computes the Hausdorff distance, uses `scipy` implementation of 'an efficient algorithm for
    calculating the exact Hausdorff distance.' provided by A. A. Taha et al.
    Args:
        ground_true ground_true (np.ndarray[bool]): ground true mask to be compared with predicted one.
        predicted predicted (np.ndarray[bool]): predicted mask, allowed values are from {True, False}.
            Should be the same dimension as `ground_true`.
    Returns:
        double: The directed Hausdorff distance.
    """
    u = np.array(np.where(truth)).T
    v = np.array(np.where(prediction)).T
    hd, _, _ = directed_hausdorff(u, v)
    return hd

