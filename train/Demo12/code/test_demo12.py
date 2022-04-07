#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("/public/siwenbin/DLQSM_YH/")

#
# Description:
#  Testing code of QSMnet 
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

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


PATCH_SIZE = (184,210,144)
QSMnet1, model_single2 = generate_multi_model(1, PATCH_SIZE, use_bn=True) 
QSMnet1.load_weights('/public/siwenbin/DLQSM_YH/Aug_test/Demo12/tmp_model/js/model_200.h5')

CASES_TEST = 12
FD_DATA = '/public/siwenbin/DLQSM_YH/Aug_test/Demo12/data/testdata/'
filename = '{0}/{1}/RDF.mat'.format(FD_DATA, CASES_TEST)
RDF = np.real(load_mat(filename, varname='RDF'))
filename = '{0}/{1}/COSMOS.mat'.format(FD_DATA, CASES_TEST)
QSM = np.real(load_mat(filename, varname='cosmos1'))

ndir = RDF.shape[-1]
RDFs = []
QSMs = []
for i_dir in range(ndir):
    rdf = RDF[...,i_dir]
    qsm = QSM[...,i_dir]
      
    RDFs.append(rdf[np.newaxis,...])
    QSMs.append(qsm[np.newaxis,...])
    
RDFs = np.concatenate(RDFs, axis=0)[..., np.newaxis]
QSMs = np.concatenate(QSMs, axis=0)[..., np.newaxis]
pred_QSMnet = QSMnet1.predict(RDFs, batch_size=1, verbose=1)

sio.savemat('/public/siwenbin/DLQSM_YH/Aug_test/Demo12/data/predQSM/Demo12_QSMnetdata_test1_dir1_hem_simulation.mat',{'predQSM':pred_QSMnet})
