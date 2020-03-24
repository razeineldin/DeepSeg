from config import *
from keras.layers import *
from keras.models import *

def get_decoder_model(input, output):
    img_input = input
    o = output
    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height * output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height * output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.encoder_name= ""
    return model

# UNet decoder
def get_unet_decoder(n_classes, encoder, input_height=224, input_width=224, depth=3, filter_size=32, encoder_name=None, up_layer=False, trainable=True):
    img_input, levels = encoder(input_height=input_height, input_width=input_width, depth=depth, filter_size=filter_size, encoder_name=encoder_name)
    [f1, f2, f3, f4, f5] = levels 
     
    # 512
    up6 = Conv2D(filter_size*2**3, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(f5))
    merge6 = concatenate([f4,up6], axis = 3)
    conv6 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge6)
    conv6 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv6)
    # 256
    up7 = Conv2D(filter_size*2**2, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([f3,up7], axis = 3)
    conv7 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge7)
    conv7 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv7)
    # 128
    up8 = Conv2D(filter_size*2, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([f2,up8], axis = 3)
    conv8 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge8)
    conv8 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv8)
    # 64
    up9 = Conv2D(filter_size, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([f1,up9], axis = 3)
    o = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge9)
    o = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(o)

    if(up_layer):
        up10 = Conv2D(filter_size, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(o))
        merge10 = concatenate([img_input,up10], axis = 3)
        o = Conv2D(int(filter_size/2), (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(merge10)
        o = Conv2D(int(filter_size/2), (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(o)

    o = Conv2D(n_classes, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(o)

    model = get_decoder_model(img_input, o)
    if (encoder_name == 'ResNet50') or (encoder_name == 'NASNetMobile') or (encoder_name == 'NasNetLarge'):
        n_pads = 2 # for one_side_pad
    elif encoder_name == 'Xception':
        n_pads = 5 # for 3 one_side_pad
    else:
        n_pads = 0 # for reshape

    if not trainable and up_layer:
        for layer in model.layers:
            layer.trainable=False   # Make the layer non-trainable
        for layer in model.layers[-26-n_pads:]:
            layer.trainable=True    # Make only the last layers trainable
    elif not trainable:
        for layer in model.layers:
            layer.trainable=False   # Make the layer non-trainable
        for layer in model.layers[-22-n_pads:]:
            layer.trainable=True    # Make only the last layers trainable
    return model

# Modified UNet decoder
def get_unet_modified_decoder(n_classes, encoder, input_height=224, input_width=224, depth=3, filter_size=64, encoder_name=None, up_layer=False, trainable=True):
    img_input, levels = encoder(input_height=input_height, input_width=input_width, depth=depth, filter_size=filter_size, encoder_name=encoder_name)
    [f1, f2, f3, f4, f5] = levels 

    # 512
    up6 = Conv2D(filter_size*2**3, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(f5))
    merge6 = concatenate([f4,up6], axis = 3)
    conv6 = Conv2D(filter_size*2**3, (3, 3), padding = 'same', data_format='channels_last')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(filter_size*2**3, (3, 3), padding = 'same', data_format='channels_last')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    # conv6 = Dropout(0.5)(conv6)

    # 256
    up7 = Conv2D(filter_size*2**2, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([f3,up7], axis = 3)
    conv7 = Conv2D(filter_size*2**2, (3, 3), padding = 'same', data_format='channels_last')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(filter_size*2**2, (3, 3), padding = 'same', data_format='channels_last')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    # 128
    up8 = Conv2D(filter_size*2, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([f2,up8], axis = 3)
    conv8 = Conv2D(filter_size*2, (3, 3), padding = 'same', data_format='channels_last')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(filter_size*2, (3, 3), padding = 'same', data_format='channels_last')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    # 64
    up9 = Conv2D(filter_size, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([f1,up9], axis = 3)
    o = Conv2D(filter_size, (3, 3), padding = 'same', data_format='channels_last')(merge9)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(filter_size, (3, 3), padding = 'same', data_format='channels_last')(merge9)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)


    if(up_layer):
        up10 = Conv2D(filter_size, (2, 2), activation = 'relu', padding = 'same', data_format='channels_last')(UpSampling2D(size = (2,2))(o))
        merge10 = concatenate([img_input,up10], axis = 3)
        o = Conv2D(int(filter_size/2), (3, 3), padding = 'same', data_format='channels_last')(merge10)
        o = BatchNormalization()(o)
        o = Activation('relu')(o)
        o = Conv2D(int(filter_size/2), (3, 3), padding = 'same', data_format='channels_last')(o)
        o = BatchNormalization()(o)
        o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), padding = 'same', data_format='channels_last')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    model = get_decoder_model(img_input, o)
    if (encoder_name == 'ResNet50') or (encoder_name == 'NASNetMobile') or (encoder_name == 'NasNetLarge'):
        n_pads = 2 # for one_side_pad
    elif encoder_name == 'Xception':
        n_pads = 5 # for 3 one_side_pad
    else:
        n_pads = 0 # for reshape

    if not trainable and up_layer:
        for layer in model.layers:
            layer.trainable=False   # Make the layer non-trainable
        for layer in model.layers[-26-n_pads:]:
            layer.trainable=True    # Make only the last layers trainable
    elif not trainable:
        for layer in model.layers:
            layer.trainable=False   # Make the layer non-trainable
        for layer in model.layers[-22-n_pads:]:
            layer.trainable=True    # Make only the last layers trainable
    return model
