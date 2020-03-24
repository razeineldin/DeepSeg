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
from keras_applications import correct_pad
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.utils import get_file
from keras import backend as K

K.set_image_data_format('channels_last')
if K.image_data_format() == 'channels_first':
    IMAGE_ORDERING = 'channels_first'
elif K.image_data_format() == 'channels_last':
    IMAGE_ORDERING = 'channels_last'

# Define models' functions
### ResNet 2D model
def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[:, :, :-1, :-1 ])(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[:, :-1, :-1, :  ])(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block_resnet50(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

### Mobilenet 2D model
def relu6(x):
    return K.relu(x, max_value=6)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel, data_format=IMAGE_ORDERING, padding='same',
            use_bias=False, strides=strides, name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3), data_format=IMAGE_ORDERING, padding='same', 
                     depth_multiplier=depth_multiplier, strides=strides,
                     use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), data_format=IMAGE_ORDERING,
            padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
 
### NASNet functions
def _separable_conv_block(ip, filters,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          block_id=None):
    '''Adds 2 blocks of [relu-separable conv-batchnorm].
    # Arguments
        ip: Input tensor
        filters: Number of output filters per layer
        kernel_size: Kernel size of separable convolutions
        strides: Strided convolution for downsampling
        block_id: String block_id
    # Returns
        A Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('separable_conv_block_%s' % block_id):
        x = Activation('relu')(ip)
        if strides == (2, 2):
            x = ZeroPadding2D(padding=correct_pad(K, x, kernel_size), name='separable_conv_1_pad_%s' % block_id)(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % block_id, padding=conv_pad, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='separable_conv_1_bn_%s' % (block_id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % block_id, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='separable_conv_2_bn_%s' % (block_id))(x)
    return x

def _adjust_block(p, ip, filters, block_id=None):
    '''Adjusts the input `previous path` to match the shape of the `input`.
    Used in situations where the output number of filters needs to be changed.
    # Arguments
        p: Input tensor which needs to be modified
        ip: Input tensor whose shape needs to be matched
        filters: Number of output filters to be matched
        block_id: String block_id
    # Returns
        Adjusted Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    img_dim = 2 if K.image_data_format() == 'channels_first' else -2

    ip_shape = K.int_shape(ip)

    if p is not None:
        p_shape = K.int_shape(p)

    with K.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p_shape[img_dim] != ip_shape[img_dim]:
            with K.name_scope('adjust_reduction_block_%s' % block_id):
                p = Activation('relu', name='adjust_relu_1_%s' % block_id)(p)
                p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_1_%s' % block_id)(p)
                p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, name='adjust_conv_1_%s' % block_id, kernel_initializer='he_normal')(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_2_%s' % block_id)(p2)
                p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, name='adjust_conv_2_%s' % block_id, kernel_initializer='he_normal')(p2)

                p = concatenate([p1, p2], axis=channel_dim)
                p = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)

        elif p_shape[channel_dim] != filters:
            with K.name_scope('adjust_projection_block_%s' % block_id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='adjust_conv_projection_%s' % block_id, use_bias=False, kernel_initializer='he_normal')(p)
                p = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)
    return p


def _normal_a_cell(ip, p, filters, block_id=None):
    '''Adds a Normal cell for NASNet-A (Fig. 4 in the paper).
    # Arguments
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id
    # Returns
        A Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('normal_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % block_id, use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='normal_bn_1_%s' % block_id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), block_id='normal_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, block_id='normal_right1_%s' % block_id)
            x1 = add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), block_id='normal_left2_%s' % block_id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), block_id='normal_right2_%s' % block_id)
            x2 = add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % (block_id))(h)
            x3 = add([x3, p], name='normal_add_3_%s' % block_id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % (block_id))(p)
            x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',name='normal_right4_%s' % (block_id))(p)
            x4 = add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, block_id='normal_left5_%s' % block_id)
            x5 = add([x5, h], name='normal_add_5_%s' % block_id)

        x = concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name='normal_concat_%s' % block_id)
    return x, ip

def _reduction_a_cell(ip, p, filters, block_id=None):
    '''Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).
    # Arguments
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id
    # Returns
        A Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('reduction_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % block_id, use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='reduction_bn_1_%s' % block_id)(h)
        h3 = ZeroPadding2D(padding=correct_pad(K, h, 3), name='reduction_pad_1_%s' % block_id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), block_id='reduction_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id='reduction_right1_%s' % block_id)
            x1 = add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='reduction_left2_%s' % block_id)(h3)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id='reduction_right2_%s' % block_id)
            x2 = add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='valid', name='reduction_left3_%s' % block_id)(h3)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), block_id='reduction_right3_%s' % block_id)
            x3 = add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % block_id)(x1)
            x4 = add([x2, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), block_id='reduction_left4_%s' % block_id)
            x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='reduction_right5_%s' % block_id)(h3)
            x5 = add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

        x = concatenate([x2, x3, x4, x5], axis=channel_dim, name='reduction_concat_%s' % block_id)
        return x, ip

### MobileNetV2
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, 3), name=prefix + 'pad')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)

    x = ReLU(6., name=prefix + 'depthwise_relu')(x)
    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

### DenseNet
def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition 
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense 
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def get_unet_encoder(input_height=224,  input_width=224, depth=3, filter_size = 32, kernel = 3, pool_size = 2, encoder_name='UNet'):
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(depth, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, depth))
    # 64
    conv1 = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(img_input)
    conv1 = Conv2D(filter_size, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128
    conv2 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool1)
    conv2 = Conv2D(filter_size*2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 256
    conv3 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool2)
    conv3 = Conv2D(filter_size*2**2, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 512
    conv4 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool3)
    conv4 = Conv2D(filter_size*2**3, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 1024
    conv5 = Conv2D(filter_size*2**4, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(pool4)
    conv5 = Conv2D(filter_size*2**4, (3, 3), activation = 'relu', padding = 'same', data_format='channels_last')(conv5)
    drop5 = Dropout(0.5)(conv5)

    return img_input, [conv1, conv2, conv3, conv4, drop5]

def get_unet_modified_encoder(input_height=224,  input_width=224, depth=3, filter_size = 32, kernel = 3, pool_size = 2, encoder_name='UNet_Modified'):
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(depth, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, depth))
    # 64
    conv1 = Conv2D(filter_size, (3, 3), padding = 'same', data_format='channels_last')(img_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filter_size, (3, 3), padding = 'same', data_format='channels_last')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128
    conv2 = Conv2D(filter_size*2, (3, 3), padding = 'same', data_format='channels_last')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(filter_size*2, (3, 3), padding = 'same', data_format='channels_last')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 256
    conv3 = Conv2D(filter_size*2**2, (3, 3), padding = 'same', data_format='channels_last')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(filter_size*2**2, (3, 3), padding = 'same', data_format='channels_last')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 512
    conv4 = Conv2D(filter_size*2**3, (3, 3), padding = 'same', data_format='channels_last')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(filter_size*2**3, (3, 3), padding = 'same', data_format='channels_last')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 1024

    conv5 = Conv2D(filter_size*2**4, (3, 3), padding = 'same', data_format='channels_last')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(filter_size*2**4, (3, 3), padding = 'same', data_format='channels_last')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    drop5 = Dropout(0.5)(conv5)

    return img_input, [conv1, conv2, conv3, conv4, drop5]

def get_vgg16_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', encoder_name='VGG16'):
    assert input_height%32 == 0
    assert input_width%32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(depth,input_height,input_width))
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height,input_width, depth))
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    f1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    f2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    f3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    f4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    f5 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
    
    if pretrained == 'imagenet':
        VGG_Weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(VGG_Weights_path)
    return img_input, [f1, f2, f3, f4, f5]

def get_resnet50_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', encoder_name='ResNet50'):
    assert input_height%32 == 0
    assert input_width%32 == 0

    if IMAGE_ORDERING == 'channels_first':
        bn_axis = 1
        img_input = Input(shape=(depth, input_height, input_width))
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, depth))
        bn_axis = 3
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    
    x = conv_block_resnet50(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block_resnet50(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x 

    x = conv_block_resnet50(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x 

    x = conv_block_resnet50(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x 

    x = AveragePooling2D((7, 7), data_format=IMAGE_ORDERING, name='avg_pool')(x)
    # f6 = x 
    
    if pretrained == 'imagenet':
        weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(weights_path)
    return img_input, [f1, f2, f3, f4, f5]

def get_mobilenet_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', alpha=1.0, encoder_name='MobileNet'):
    assert (K.image_data_format() == 'channels_last'), "Currently only channels last mode is supported"
    assert (IMAGE_ORDERING == 'channels_last'), "Currently only channels last mode is supported"
    assert (input_height == 224), "For mobilenet, 224 input_height is supported "
    assert (input_width == 224), "For mobilenet, 224 width is supported "
    assert input_height%32 == 0
    assert input_width%32 == 0
    
    depth_multiplier=1
    dropout=1e-3

    img_input = Input(shape=(input_height, input_width, depth))

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1) 
    f1 = x
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)  
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3) 
    f2 = x
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)  
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5) 
    f3 = x
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6) 
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7) 
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8) 
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9) 
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10) 
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11) 
    f4 = x 
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)  
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13) 
    f5 = x 

    if pretrained == 'imagenet' :
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
        BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weight_path)
        Model(img_input, x).load_weights(weights_path)

    return img_input, [f1, f2, f3, f4, f5 ]

def get_xception_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', encoder_name='Xception'):
    if IMAGE_ORDERING == 'channels_first':
        channel_axis = 1
        img_input = Input(shape=(depth, input_height, input_width))
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_th_dim_ordering_th_kernels_notop.h5"
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, depth))
        channel_axis = 3
        pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)
    f1 = one_side_pad(ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)) # F1: (?, 109, 109, 64)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)
    f2 = one_side_pad(x) # F2: (?, 55, 55, 128) 

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)
    f3 = x 

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])
    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)
    f4 = x

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    f5 = x

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    if pretrained == 'imagenet':
        weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)
        Model(img_input, x).load_weights(weights_path)
    return img_input, [f1, f2, f3, f4, f5]

def get_nasnet_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', encoder_name='NASNetMobile'):
    # NASNetMobile = 224, NASNetLarge = 331, NOTE: Only "channels_last" is supported
    if encoder_name=='NASNetMobile':
        penultimate_filters=1056
        num_blocks=4
        stem_block_filters=32
        skip_reduction=False
        filter_multiplier=2
    elif encoder_name=='NASNetLarge':
        penultimate_filters=4032
        num_blocks=6
        stem_block_filters=96
        skip_reduction=True
        filter_multiplier=2

    img_input = Input(shape=(input_height, input_width, depth))
    channel_dim = 3
    filters = penultimate_filters // 24

    x = Conv2D(stem_block_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='stem_conv1', kernel_initializer='he_normal')(img_input)
    x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)
    f1 = one_side_pad(x)

    p = None
    x, p = _reduction_a_cell(x, p, filters // (filter_multiplier ** 2), block_id='stem_1')
    f2 = x
    x, p = _reduction_a_cell(x, p, filters // filter_multiplier, block_id='stem_2')

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id='%d' % (i))
    f3 = x

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier, block_id='reduce_%d' % (num_blocks))
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier, block_id='%d' % (num_blocks + i + 1))
    f4 = x

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier ** 2, block_id='reduce_%d' % (2 * num_blocks))
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier ** 2, block_id='%d' % (2 * num_blocks + i + 1))

    x = Activation('relu')(x)
    f5 = x
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    if pretrained == 'imagenet':
        BASE_WEIGHTS_PATH = ('https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/')
        NASNET_MOBILE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-mobile-no-top.h5'
        NASNET_LARGE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-large-no-top.h5'
        if encoder_name=='NASNetMobile':
            weights_path = get_file('nasnet_mobile_no_top.h5', NASNET_MOBILE_WEIGHT_PATH_NO_TOP)
        elif encoder_name=='NASNetLarge':
            weights_path = get_file('nasnet_large_no_top.h5', NASNET_LARGE_WEIGHT_PATH_NO_TOP)
        
        Model(img_input, x).load_weights(weights_path)
    return img_input, [f1, f2, f3, f4, f5]

def get_mobilenetv2_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', alpha=1.0, encoder_name='MobileNetV2'):
    assert (K.image_data_format() == 'channels_last'), "Currently only channels last mode is supported"
    assert (IMAGE_ORDERING == 'channels_last'), "Currently only channels last mode is supported"
    assert (input_height == 224), "For mobilenet, 224 input_height is supported "
    assert (input_width == 224), "For mobilenet, 224 width is supported "
    assert input_height%32 == 0
    assert input_width%32 == 0
    
    pooling=None
    rows = input_height
    cols = input_width
    channel_axis = -1
    img_input = Input(shape=(input_height, input_width, depth))

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(K, img_input, 3), name='Conv1_pad')(img_input)
    x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False, name='Conv1')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = ReLU(6., name='Conv1_relu')(x)
    # f1 = x # F1: (?, 112, 112, 32)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    f1 = x # F1: (?, 112, 112, 16)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    f2 = x # F2: (?, 56, 56, 24)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)
    f3 = x # F3: (?, 28, 28, 32)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)
    # f4 = x # F4: (?, 14, 14, 64)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)
    f4 = x # F4: (?, 14, 14, 96)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
    # f4 = x # F4: (?, 7, 7, 160)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)
    f5 = x # F5: (?, 7, 7, 1280)
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)
    
    if pretrained == 'imagenet' :
        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
        BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/')
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weight_path, cache_subdir='models')
  
        Model(img_input, x).load_weights(weights_path)
    return img_input, [f1, f2, f3, f4, f5 ]

def get_densenet121_encoder(input_height=224, input_width=224, depth=3, filter_size = 64, pretrained='imagenet', encoder_name='DenseNet121'):
    assert (K.image_data_format() == 'channels_last'), "Currently only channels last mode is supported"
    assert (IMAGE_ORDERING == 'channels_last'), "Currently only channels last mode is supported"

    if encoder_name=='DenseNet121':
        blocks = [6, 12, 24, 16]
    elif encoder_name=='DenseNet169':
        blocks = [6, 12, 32, 32]
    elif encoder_name=='DenseNet201':
        blocks = [6, 12, 48, 32]
    else:
        blocks = [6, 12, 24, 16]

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    img_input = Input(shape=(input_height, input_width, depth))

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    f1 = x
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    f2 = x
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    f3 = x
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    f4 = x
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)
    f5 = x

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)

    if pretrained == 'imagenet':
        BASE_WEIGTHS_PATH = ('https://github.com/keras-team/keras-applications/releases/download/densenet/')
        DENSENET121_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
        DENSENET169_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
        DENSENET201_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
 
        if blocks == [6, 12, 24, 16]:
            weights_path = get_file('densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET121_WEIGHT_PATH_NO_TOP)
        elif blocks == [6, 12, 32, 32]:
            weights_path = get_file('densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET169_WEIGHT_PATH_NO_TOP)
        elif blocks == [6, 12, 48, 32]:
            weights_path = get_file('densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5', DENSENET201_WEIGHT_PATH_NO_TOP)
        Model(img_input, x).load_weights(weights_path)

    return img_input, [f1, f2, f3, f4, f5 ]
