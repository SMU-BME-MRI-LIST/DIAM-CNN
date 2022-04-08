import tensorflow as tf
from keras.layers import ZeroPadding3D
from keras.engine import Layer

import sys
sys.path.append('..')
from utils import dipole_kernel

#
Padding method of edge pixels
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

class MirrorPadding3D(ZeroPadding3D):
    '''
    Pad input in a reflective manner
    '''
    def call(self, x):
        pattern = [[0, 0],
                   list(self.padding[0]),
                   list(self.padding[1]),
                   list(self.padding[2]),
                   [0, 0]]
        return tf.pad(x, pattern, mode='SYMMETRIC')
    
    

class DipoleConv(Layer):
    """
    Dipole Convolution layer, non-trainable
    """
    def __init__(self, voxel_size=(1,1,1),
                       B0_dir=(0,0,1),
                       space='kspace', 
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.voxel_size = voxel_size
        self.B0_dir = B0_dir
        self.space = space
        
        return
    
        
    def build(self, input_shape):
        nchan = input_shape[-1]
        assert nchan == 1, '# Channel >1 not supported'
        kernel_shape = input_shape[-4:-1]
        
        # prepare weight
        #   dipole kernel
        kernel = dipole_kernel(kernel_shape, self.voxel_size, self.B0_dir)
        
        # add weight
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel.shape,
                                      initializer='uniform',
                                      trainable=False)
        self.set_weights([kernel])
        self.use_bias = False
        self.bias = None
        
        super().build(input_shape)
        
        return
    
    
    def call(self, x):
        x_cplx = tf.cast(x[..., 0], tf.complex64)
        kernel_cplx = tf.cast(self.kernel[tf.newaxis, ...], tf.complex64)
        dx_cplx = tf.ifft3d(tf.fft3d(x_cplx)*kernel_cplx)
        dx = tf.real(dx_cplx)[..., tf.newaxis]
        return dx
    
    
    def compute_output_shape(self, input_shape):
        return input_shape
