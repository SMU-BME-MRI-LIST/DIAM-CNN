# Definition of
#   3D UNet
# Author: Zhe Liu (zl376@cornell.edu)
# Modified by: Jinwei Zhang (jz853@cornell.edu)
# Date: 2018-08-27

from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import MaxPooling3D
from keras.layers import UpSampling3D
from keras.layers import Cropping3D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv3DTranspose
from keras.layers import Concatenate
from keras.models import Model
from .customized_layer import MirrorPadding3D
from keras.layers.core import Lambda


def UNet_3d555(img_size, filter_base=32,
                      kernel_size=(5,5,5),
                      level=5,
                      use_bn=True,
                      use_deconv=True, 
                      multi_input=False,
                      multi_output=False,
                      bilateral_structure=False, 
                      multi_output2=False):
                      
    print(img_size)                                    
    if multi_input:
        inputs = Input(img_size + (3,), name='input')
        x = Lambda(lambda x: x[..., 0:3])(inputs)
        x_ = Lambda(lambda x: x[..., 3:4])(inputs)
        print(x.shape)
        print('-----------------------------------------------------------------------')
        
        
    else:
        inputs = Input(img_size + (1,), name='input')
        x = inputs
    
    filter_sizes = [ min(512, filter_base * 2**i) for i in range(level) ]
    endpoints = []

    # down-stream
    for i in range(level-1):
        filter_size = filter_sizes[i]
        name_template = '{{}}_{0}d'.format(i+1)

        # Conv-Act-BN
        x = conv_act_block(x, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)
        endpoints.append(x)

        # Down-sampling
        x = MaxPooling3D(pool_size=(2,2,2), padding='same', name=name_template.format('down'))(x)

    # last level
    filter_size = filter_sizes[level-1]
    name_template = '{{}}_{0}'.format(level)
    # Conv-Act-BN
    x = conv_act_block(x, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)
    x1 = x
    x2 = x
    
    # up-stream
    for i in range(level-2, -1, -1):
        filter_size = filter_sizes[i]
        name_template = '{{}}_{0}u'.format(i+1)

        # Up-sampling
        if use_deconv:
            x1 = Conv3DTranspose(filter_size, (2,2,2), strides=2, padding='valid', kernel_initializer='he_normal', name=name_template.format('deconv'))(x1)
            #x1 = Activation('relu', name=name_template.format('actdc'))(x1)
            #if use_bn:
            #    x1 = BatchNormalization(name=name_template.format('bndc'))(x1)            
        else:
            x1 = UpSampling3D(size=(2,2,2), name=name_template.format('up'))(x1)
        # Concatenation (crop if needed)
        y = endpoints[i]
        shape_x1, shape_y = x1._keras_shape[1:-1], y._keras_shape[1:-1]
        cropping = tuple( ((i-j)//2, (i-j+1)//2) for i,j in zip(shape_x1, shape_y) )
        x1 = Cropping3D(cropping=cropping)(x1)
        x1 = Concatenate(axis=-1, name=name_template.format('cat'))([x1, y])

        # Conv-Act-BN
        x1 = conv_act_block(x1, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)
    
    # bilateral structure
    if bilateral_structure:
        # up-stream
        for i in range(level-2, -1, -1):
            filter_size = filter_sizes[i]
            name_template = '{{}}_{0}u'.format(i+1+level)

            # Up-sampling
#             if use_deconv:
#                 x2 = Conv3DTranspose(filter_size, kernel_size, strides=2, padding='valid', kernel_initializer='he_normal', name=name_template.format('deconv'))(x2)
#                 x2 = Activation('relu', name=name_template.format('actdc'))(x2)
#                 if use_bn:
#                     x2 = BatchNormalization(name=name_template.format('bndc'))(x2)            
#             else:
            x2 = UpSampling3D(size=(2,2,2), name=name_template.format('up'))(x2)
            # Concatenation (crop if needed)
            y = endpoints[i]
            shape_x2, shape_y = x2._keras_shape[1:-1], y._keras_shape[1:-1]
            cropping = tuple( ((i-j)//2, (i-j+1)//2) for i,j in zip(shape_x2, shape_y) )
            x2 = Cropping3D(cropping=cropping)(x2)
            x2 = Concatenate(axis=-1, name=name_template.format('cat'))([x2, y])

            # Conv-Act-BN
            x2 = conv_act_block(x2, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)
    
    if bilateral_structure:
        output1 = Conv3D(1, kernel_size=(1,1,1), kernel_initializer='he_normal', name='conv_final1')(x1)
        if multi_output2:
            output2 = Conv3D(4, kernel_size=(1,1,1), kernel_initializer='he_normal', name='conv_final2')(x2)
        else:
            output2 = Conv3D(1, kernel_size=(1,1,1), kernel_initializer='he_normal', name='conv_final2')(x2)
        if multi_input:
            outputs_ = Concatenate(axis=-1, name='cat_final_')([output1, output2])
            outputs = Concatenate(axis=-1, name='cat_final')([outputs_, x_])
        else:
            outputs = Concatenate(axis=-1, name='cat_final')([output1, output2])
    elif multi_output:
        outputs = Conv3D(2, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', name='conv_final')(x1)
    else:
        outputs = Conv3D(1, kernel_size=(1,1,1), kernel_initializer='he_normal', name='conv_final')(x1)

    model = Model(inputs=inputs, outputs=outputs)
    return model



def conv_act_block(x, 
                   filter_size, kernel_size, 
                   use_bn=True, 
                   name_template=''):
    padding = tuple( x//2 for x in kernel_size )
    x = Conv3D(filter_size, kernel_size=kernel_size, kernel_initializer='he_normal', name=name_template.format('conv1'))(MirrorPadding3D(padding)(x))
    if use_bn:
        x = BatchNormalization(name=name_template.format('bn1'))(x)
    x = Activation('relu', name=name_template.format('act1'))(x)

    x = Conv3D(filter_size, kernel_size=kernel_size, kernel_initializer='he_normal', name=name_template.format('conv2'))(MirrorPadding3D(padding)(x))
    if use_bn:
        x = BatchNormalization(name=name_template.format('bn2'))(x)
    x = Activation('relu', name=name_template.format('act2'))(x)
    return x

def N_std_identity(args):
    N_std = args
    return N_std