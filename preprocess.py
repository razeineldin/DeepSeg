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

import shutil
import SimpleITK as sitk
from nipype.interfaces.ants import N4BiasFieldCorrection

config["modalities"] = ["flair", "t1", "t1ce", "t2"]

def correct_bias(in_path, out_path, image_type=sitk.sitkFloat64):
    # N. Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_path
    correct.inputs.output_image = out_path
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_path, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_path)
        return os.path.abspath(out_path)

def get_image_path(subject_folder, name):
    file_name = os.path.join(subject_folder, "*" + name + ".nii.gz")
    return glob.glob(file_name)[0]

def check_origin(in_path, in_path2):
    image = sitk.ReadImage(in_path)
    image2 = sitk.ReadImage(in_path2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_path)

def normalize_image(in_path, out_path, bias_correction=True):
    if bias_correction:
        correct_bias(in_path, out_path)
    else:
        shutil.copy(in_path, out_path)

def preprocess_brats_folder(in_folder, out_folder, truth_name='seg', no_bias_correction_modalities=None):
    for name in config["modalities"]:
        image_image = get_image_path(in_folder, name)
        case_ID = os.path.basename(out_folder)
        out_path = os.path.abspath(os.path.join(out_folder, "%s_%s.nii.gz"%(case_ID, name)))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_image, out_path, bias_correction=perform_bias_correction)

    truth_image = get_image_path(in_folder, truth_name)
    out_path = os.path.abspath(os.path.join(out_folder, "%s_truth.nii.gz"%(case_ID)))
    shutil.copy(truth_image, out_path)
    check_origin(out_path, get_image_path(in_folder, config["modalities"][0])) # check with the flair image

def preprocess_brats_data(brats_folder, out_folder, overwrite=False, no_bias_correction_modalities=("flair")):
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                preprocess_brats_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities)

def main(brats_path, preprocessed_brats):
    preprocess_brats_data(brats_path, preprocessed_brats)

if __name__ == "__main__":
    main(brats_path=config['brats_path'], preprocessed_brats=config['preprocessed_brats'])
