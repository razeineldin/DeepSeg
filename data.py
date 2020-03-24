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
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm
from cv2 import imread, resize, INTER_NEAREST

seq = iaa.Sequential([
    iaa.Fliplr(config['flip_H']), # horizontally flip 20% of all images
    iaa.Flipud(config['flip_V']), # vertically flip 20% of all images

    iaa.Affine(scale={"x": config['scale_X'], "y": config['scale_Y']}, 
        translate_percent={"x": config['translate_X'], "y": config['translate_Y']}, # shift 
        rotate=config['rotate'], 
        shear=config['shear'])
    #iaa.ElasticTransformation(config['elastic'])
], random_order=config['random_order']) # apply augmenters in random order

def get_augment_seg(img, seg, n_classes):    
    aug_det = seq.to_deterministic() 
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=n_classes, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    return image_aug, segmap_aug

def get_pairs_from_paths(images_path, segs_path):
    images = glob.glob(os.path.join(images_path, config['train_modality'][0] + "*.png"))
    segmentations = glob.glob(os.path.join(segs_path, "*.png")) 
    ret = []
    for im in images:
        seg_bnme = os.path.basename(im)
        seg = os.path.join(segs_path, seg_bnme)
        ret.append((im, seg))
    return ret

def get_images_arr(path, width, height, imgNorm="norm", odering='channels_first', train_modalities=config['train_modality']):
    imgs = np.zeros((width, height, len(train_modalities)))
    img_name = os.path.basename(path)
    train_dir = os.path.dirname(os.path.dirname(path))

    for i in range(len(train_modalities)):
        im = os.path.join(train_dir, train_modalities[i]+ img_name)
        im = imread(im, 0) # load as a grayscale image
        im = resize(im, (width, height), interpolation = INTER_NEAREST)
        imgs[:,:,i] = im

    if imgNorm == "sub_and_divide":
        imgs = np.float32(resize(imgs, (width, height), interpolation = INTER_NEAREST)) / 127.5 - 1
    elif imgNorm == "divide":
        imgs = resize(imgs, (width, height), interpolation = INTER_NEAREST)
        imgs = imgs.astype(np.float32)
        imgs = imgs/255.0
    elif imgNorm == "norm":
        # Intensity normalisation for each modality (zero mean and unit variance)
        for i in range(len(train_modalities)):
            img = imgs[:,:,i]
            # img = resize(img, (width, height), interpolation = INTER_NEAREST)
            img_mean = img.mean()
            img_std = img.std()
            if(img_std != 0):
                imgs[:,:,i] = (img - img_mean) / img_std
            else:
                # print('Error!!: invalid value encountered in true_divide')
                imgs[:,:,i] = (img - img_mean)

    if odering == 'channels_first':
        imgs = np.rollaxis(imgs, 2, 0)
    return imgs

def get_image_arr(path, width, height, imgNorm="norm", odering='channels_first'):
    if type(path) is np.ndarray:
        img = path
    else:
        # Read grayscale image as (width, height, 3)
        img = imread(path, 1)
        # Read grayscale image as (width, height, 1)
        # img = imread(path, 0)
        # img = np.reshape(width, height, 1)

    if imgNorm == "sub_and_divide":
        img = np.float32(resize(img, (width, height), interpolation = INTER_NEAREST)) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = resize(img, (width, height))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = img[ :, :, ::-1 ]
    elif imgNorm == "divide":
        img = resize(img, (width, height), interpolation = INTER_NEAREST)
        img = img.astype(np.float32)
        img = img/255.0
    elif imgNorm == "norm":
        # Intensity normalization (zero mean and unit variance)
        img = resize(img, (width, height), interpolation = INTER_NEAREST)
        img_mean = img.mean()
        img_std = img.std()
        # img = (img - img_mean) / img_std
        if(img_std != 0):
            img = (img - img_mean) / img_std
        else:
            # print('Error!!: invalid value encountered in true_divide')
            img = (img - img_mean)

    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def get_segmentation_arr(path, classes, width, height):
    n_classes = len(classes)
    seg_labels = np.zeros((height, width, n_classes))
        
    if type(path) is np.ndarray:
        img = path
    else:
        img = imread(path, 1)

    img = resize(img, (width, height), interpolation=INTER_NEAREST)
    img = img[:, :, 0]

    # change the predicted label 4 to value of 3 (for the data augmentation)
    img[img==4] = 3
    for c in range(n_classes):
        seg_labels[:, :, c ] = (img == c).astype(int)

    seg_labels = np.reshape(seg_labels, (width*height, n_classes))
    return seg_labels

def verify_segmentation_dataset(images_path, segs_path, n_classes):
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    assert len(img_seg_pairs)>0, "Dataset looks empty or path is wrong "
    
    for im_fn, seg_fn in tqdm(img_seg_pairs) :
        img = imread(im_fn)
        seg = imread(seg_fn)
        assert (img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1]), "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
    print("Dataset verified! ")

def image_segmentation_generator(images_path, segs_path,  batch_size,  classes, input_height, input_width, output_height, output_width, do_augment=False, shuffle=True):
    n_classes = len(classes)
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    
    if shuffle: random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)
    while True:
        X = []
        Y = []
        for _ in range(batch_size) :
            im, seg = next(zipped) 
            seg = imread(seg, 1)

            n_train_modality = len(config['train_modality'])
            if(n_train_modality<2):
                img = get_image_arr(im, input_width, input_height,odering=IMAGE_ORDERING)
                if do_augment:
                    img, seg[:,:,0] = get_augment_seg(img, seg[:,:,0], n_classes)
                X.append(img)
            else:
                imgs = get_images_arr(im, input_width, input_height,odering=IMAGE_ORDERING)
                if do_augment:
                    imgs, seg[:,:,0] = get_augment_seg(imgs, seg[:,:,0], n_classes)
                X.append(imgs)

            Y.append(get_segmentation_arr(seg, classes, output_width, output_height))
        yield np.array(X), np.array(Y)
