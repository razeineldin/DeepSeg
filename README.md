# DeepSeg: deep neural network framework for automatic brain tumor segmentation using magnetic resonance FLAIR images

**[2021.12.21] Update:** This paper describes our contribution to the BraTS 2021 Challenge [Ensemble CNN Networks for GBM Tumors Segmentation using Multi-parametric MRI](https://arxiv.org/abs/2112.06554) and the docker image for reproducing our predictions on the BraTS 2021 is [online](https://hub.docker.com/r/razeineldin/deepseg21)

## Prerequisites
1. Download the BRATS 2019 data by following the steps outlined on the [BRATS 2019 competition page](https://www.med.upenn.edu/cbica/brats2019/registration.html)

2. Install the following packages:
* cv2
* glob
* keras
* matplotlib
* nibabel
* numpy
* pandas
* scipy
* sklearn
* tqdm

The following packages are needed for pre-processing only:
* shutil 
* SimpleITK
* [ANTs](https://github.com/ANTsX/ANTs)

## Configuration
[config.py](config.py) is the main file used for project configuration and setup. Please read it carefully and update the paths and other setup parameters.


## Pre-procsessing
1. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs 
binaries to the PATH environmental variable. Then, perform image wise Normalization and Bias correction (using ANTs N4BiasFieldCorrection):
```
$ python3 preprocess.py
```

2. Divide the processed data into folders for different mri modalities (t1, t1ce, flair, t2)

3. Convert into 2D images
```
$ python3 preprocess_2d_images.py
```

4. Make the dataset folder and divide the images into train/valid as the following: 
``` dataset_brats19/
├── train_images
│   └── image_FLAIR
├── train_segmentation
│   ├── truth
│   └── truth_complete
├── val_images
│   └── image_FLAIR
└── val_segmentation
    ├── truth
    └── truth_complete
```

## Training
You can use our [pre-trained weights](https://drive.google.com/file/d/1oJQyfe5NHrym9Qk7Wg4nb3_Gvcf97tAN/view?usp=sharing) to get the same results as in our paper.
Or you can run the training using: 
```
$ python3 train.py
```

## Prediction
Get nifti predictions of all images in the validation directory:
```
$ python3 predict.py
```

## Evaluation
Run the evaluation using: 
```
$ python3 evaluate.py
```

## License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Citation
The work has been published in the International Journal of Computer Assisted Radiology and Surgery (IJCARS). If you find this code usefull, feel free to use it (or part of it) in your project and please cite the following paper:

    @article{Zeineldin_2020,
       title={DeepSeg: deep neural network framework for automatic brain tumor segmentation using magnetic resonance FLAIR images},
       ISSN={1861-6429},
       url={http://dx.doi.org/10.1007/s11548-020-02186-z},
       DOI={10.1007/s11548-020-02186-z},
       journal={International Journal of Computer Assisted Radiology and Surgery},
       publisher={Springer Science and Business Media LLC},
       author={Zeineldin, Ramy A. and Karar, Mohamed E. and Coburger, Jan and Wirtz, Christian R. and Burgert, Oliver},
       year={2020},
       month={May}
    }
    
## References
1. Image augmentation for machine learning experiments. https://github.com/aleju/imgaug
2. Image Segmentation Keras: https://github.com/divamgupta/image-segmentation-keras 
