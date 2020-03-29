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
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard

def train_deepseg_model(model, train_images, train_annotations, input_height=224, 
        input_width=224, output_height=224, output_width=224, classes=None, n_classes=None, 
        n_modalities=1, verify_dataset=True, epochs = 35, initial_epoch = 0, batch_size = 16, 
        validate=False, val_images=None, val_annotations=None, val_batch_size=16, 
        steps_per_epoch=512, validation_steps=200, do_augment=False):

    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying validation dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)
  
    train_gen = image_segmentation_generator(train_images, train_annotations,  batch_size, classes, input_height, input_width, output_height, output_width, do_augment)

    # callback functions
    model_checkpoint = ModelCheckpoint(config['model_checkpoints']+".{epoch:03d}-{val_dice_argmax:.2f}.hdf5", monitor='val_dice_argmax', save_best_only=False, save_weights_only=True)
    csv_logger = CSVLogger(os.path.join(config['log_dir'], config['project_name'] + '.txt'), separator=',', append=True)
    tensor_board = TensorBoard(config['tensorboard_path'])
    #model_earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto')

    if validate:
        val_gen = image_segmentation_generator(val_images, val_annotations,  val_batch_size, classes, input_height, input_width, output_height, output_width, do_augment=False, shuffle=False)
        results = model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=validation_steps,
                                      epochs=epochs, initial_epoch=initial_epoch, callbacks=[csv_logger, model_checkpoint, csv_logger, tensor_board])#, model_earlystopping])
    else:
        results = model.fit_generator(train_gen, steps_per_epoch, epochs=epochs, initial_epoch=initial_epoch, callbacks=[model_checkpoint, csv_logger, tensor_board]) #, model_earlystopping])

    return results

def main():
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

    # start training
    history = train_deepseg_model(
            unet_2d_model,
            train_images = config['train_images'],
            train_annotations = config['train_annotations'],
            input_height = config['input_height'], 
            input_width = config['input_width'],
            output_height = config['output_height'], 
            output_width = config['output_width'],
            classes = config['classes'],
            n_classes = config['n_classes'],
            verify_dataset = config['verify_dataset'],
            epochs = config['epochs'],
            initial_epoch = config['initial_epoch'],
            batch_size = config['batch_size'],
            validate = config['validate'], 
            val_images = config['val_images'], 
            val_annotations = config['val_annotations'],
            val_batch_size = config['val_batch_size'], 
            steps_per_epoch = config['steps_per_epoch'],
            validation_steps = config['validation_steps'],
            do_augment=config['do_augment']
    )

if __name__ == "__main__":
    main()
