#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Description:
#  Testing code of DIAM-CNN
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

import sys
sys.path.append("/public/siwenbin/DLQSM_YH/")

from utils import *
from callback import *
from loss import *
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, BatchNormalization, regularizers, Dropout, UpSampling2D, UpSampling3D
from keras.layers import Concatenate, MaxPooling2D, MaxPooling3D, LocallyConnected1D, Add, merge, Activation
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.models import Model, Sequential
from keras import metrics
from keras.datasets import mnist
#from skimage.measure import structural_similarity as ssim
from keras.optimizers import *
from keras.utils import multi_gpu_model
from keras.layers.advanced_activations import LeakyReLU
import scipy.io
import scipy.io as sio
import h5py

# sess = tf.Session()
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# K.set_session(sess)
import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imread, imresize
import scipy.misc
from scipy.stats import norm
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import load_model
import math
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from IPython.display import clear_output
from model.customized_layer import MirrorPadding3D
from model.common import generate_multi_model
from keras.utils import np_utils


PATCH_SIZE = (176,176,160)
QSMnet, model_single2 = generate_multi_model(1, PATCH_SIZE, use_bn=True,multi_input = True) 
QSMnet.load_weights('/public/siwenbin/DLQSM_YH/Aug_test/Demo14/tmp_model/js/model_200.h5')

CASES_TEST = 9
FD_DATA = '/public/siwenbin/DLQSM_YH/Aug_test/Demo12/data/testdata/'
filename = '{0}/{1}/RDF.mat'.format(FD_DATA, CASES_TEST)
RDF = np.real(load_mat(filename, varname='RDF1'))
filename = '{0}/{1}/COSMOS.mat'.format(FD_DATA, CASES_TEST)
QSM = np.real(load_mat(filename, varname='cosmos1'))
filename = '{0}/{1}/Mask.mat'.format(FD_DATA, CASES_TEST)
Mask = np.real(load_mat(filename, varname='Mask'))
VOXEL_SIZE = (1, 1, 1)

dipole_size = RDF.shape[:-1]
print(dipole_size)###
D = dipole_kernel(dipole_size, VOXEL_SIZE, [0, 0, 1])
Mask_D1 = abs(D)>0.3
print(Mask_D1.shape)###
Mask_D2 = (abs(D)<=0.3)

ndir = RDF.shape[-1]
RDFs = []
RDFs1 =[]
RDFs2 =[]
QSMs = []
for i_dir in range(ndir):
    rdf = RDF[...,i_dir]
    qsm = QSM[...,i_dir]
      
    RDF_D1 = np.real(np.fft.ifftn(Mask_D1*np.fft.fftn(rdf)))*(abs(qsm)> 1e-8)
    RDF_D2 = np.real(np.fft.ifftn(Mask_D2*np.fft.fftn(rdf)))*(abs(qsm)> 1e-8)
    RDF_dir = rdf
    RDFs.append(RDF_dir[np.newaxis,...])
    RDF_D1_dir = RDF_D1
    RDFs1.append(RDF_D1_dir[np.newaxis,...])
    RDF_D2_dir = RDF_D2
    RDFs2.append(RDF_D2_dir[np.newaxis,...])
    QSMs.append(qsm[np.newaxis,...])
    
RDFs = np.concatenate(RDFs, axis=0)[..., np.newaxis]
RDFs1 = np.concatenate(RDFs1, axis=0)[..., np.newaxis]
RDFs2 = np.concatenate(RDFs2, axis=0)[..., np.newaxis]
RDFs_multi_channels = np.concatenate((RDFs,RDFs1,RDFs2),axis=-1);
QSMs = np.concatenate(QSMs, axis=0)[..., np.newaxis]
pred_QSMnet = QSMnet.predict(RDFs_multi_channels, batch_size=1, verbose=1)

index_dir = 0;
fig = plt.figure()
plt.imshow(QSMs[index_dir,:,:,80,0], cmap='Greys_r', clim=(-0.15, 0.15))
fig = plt.figure()
plt.imshow(pred_QSMnet[index_dir,:,:,80,0], cmap='Greys_r', clim=(-0.15, 0.15))
               
sio.savemat('/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/predQSM/Demo14_QSMnetdata_test1_dir1_hem_simulation.mat',{'predQSM':pred_QSMnet})

