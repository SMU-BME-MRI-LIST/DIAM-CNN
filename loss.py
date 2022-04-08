import numpy as np
import tensorflow as tf
from utils import dipole_kernel
from utils import dxp, dyp, dzp

#
#Description:
#Loss function for training QSMnet and DIAM-CNN
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

def l2_loss(y_true, y_pred):
#     mask = tf.cast(abs(y_true) > 1e-5, tf.float32)
    mask = tf.ones(tf.shape(y_pred))
    loss = tf.reduce_mean(mask*tf.square(y_true - y_pred), axis=(0,1,2,3,4))
    return loss


def l1_loss(y_true, y_pred):
#     mask = tf.cast(abs(y_true) > 1e-5, tf.float32)
    mask = tf.ones(tf.shape(y_pred))#
    loss = tf.reduce_mean(mask*tf.abs(y_true - y_pred), axis=(0,1,2,3,4))
    return loss


def gradient_loss(y_true, y_pred):
#     mask = tf.cast(abs(y_true) > 1e-5, tf.float32)
    mask = tf.ones(tf.shape(y_pred))
    _, nx, ny, nz, _ = y_pred.get_shape().as_list()
    D = dipole_kernel([nx, ny, nz], [1, 1, 3], [0, 0, 1])[np.newaxis, ...]
    D = tf.convert_to_tensor(D, np.complex64)
    y_true_cplx = tf.cast(y_true[..., 0], tf.complex64)
    y_pred_cplx = tf.cast(y_pred[..., 0], tf.complex64)
    y_true_RDF = tf.ifft3d(tf.fft3d(y_true_cplx)*D)
    y_pred_RDF = tf.ifft3d(tf.fft3d(y_pred_cplx)*D)
    loss = tf.reduce_mean(mask*abs(dxp(y_true) - dxp(y_pred)), axis=(0,1,2,3,4)) + \
           tf.reduce_mean(mask*abs(dyp(y_true) - dyp(y_pred)), axis=(0,1,2,3,4)) + \
           tf.reduce_mean(mask*abs(dzp(y_true) - dzp(y_pred)), axis=(0,1,2,3,4)) + \
           tf.reduce_mean(mask[...,0]*abs(dxp(y_true_RDF) - dxp(y_pred_RDF)), axis=(0,1,2,3)) + \
           tf.reduce_mean(mask[...,0]*abs(dyp(y_true_RDF) - dyp(y_pred_RDF)), axis=(0,1,2,3)) + \
           tf.reduce_mean(mask[...,0]*abs(dzp(y_true_RDF) - dzp(y_pred_RDF)), axis=(0,1,2,3)) 
    return loss


def dipole_loss(y_true, y_pred):
#     mask = tf.cast(abs(y_true) > 1e-5, tf.float32)
    mask = tf.ones(tf.shape(y_pred))
    _, nx, ny, nz, _ = y_pred.get_shape().as_list()
    D = dipole_kernel([nx, ny, nz], [1, 1, 3], [0, 0, 1])[np.newaxis, ...]
    D = tf.convert_to_tensor(D, np.complex64)
    y_true = tf.cast(y_true[..., 0], tf.complex64)
    y_pred = tf.cast(y_pred[..., 0], tf.complex64)
    diff = tf.abs(tf.ifft3d((tf.fft3d(y_true) - tf.fft3d(y_pred))*D))
    loss = tf.reduce_mean(mask[...,0]*diff, axis=(0,1,2,3))
    return loss

def feedback_loss(y_true, y_pred):
    mask = tf.cast(abs(y_true) > 1e-5, tf.float32)
    return 0
    



# generate aggregated loss 
def generate_loss(weight_l2=0,
                  weight_l1=1,
                  weight_gradient=0.1,
                  weight_dipole=0.5):
    
    l2 = lambda x,y: l2_loss(x,y)
    l1 = lambda x,y: l1_loss(x,y)
    gradient = lambda x,y: gradient_loss(x,y)
    dipole = lambda x,y: dipole_loss(x,y)
    
    func_loss = lambda x,y: sum([ loss(x,y) * w
                                  for w, loss in zip([weight_l2, weight_l1, weight_gradient, weight_dipole], 
                                                     [l2, l1, gradient, dipole]) 
                                  if w > 0 ])
    return func_loss
    
def generate_loss1(y_true, y_pred):#
    weight_l2=0
    weight_l1=1
    weight_gradient=0.1
    weight_dipole=0.5
    func_loss = weight_l2 * l2_loss(y_true, y_pred) + weight_l1 * l1_loss(y_true, y_pred) + weight_gradient * gradient_loss(y_true, y_pred) + weight_dipole * dipole_loss(y_true, y_pred)
    return func_loss
    
def generate_loss555(y_true, y_pred):#
    weight_l2=0
    weight_l1=1
    weight_gradient=0.1
    weight_dipole=1
    func_loss = weight_l2 * l2_loss(y_true, y_pred) + weight_l1 * l1_loss(y_true, y_pred) + weight_gradient * gradient_loss(y_true, y_pred) + weight_dipole * dipole_loss(y_true, y_pred)
    return func_loss
    
def Demo22_generate_loss1(y_true, y_pred):
    weight_l2=0
    weight_l1=1
    weight_gradient=0
    weight_dipole=0
    func_loss = weight_l2 * l2_loss(y_true, y_pred) + weight_l1 * l1_loss(y_true, y_pred) + weight_gradient * gradient_loss(y_true, y_pred) + weight_dipole * dipole_loss(y_true, y_pred)
    return func_loss    
    
    
    
    
    
    
    
    
