# -*- coding: utf-8 -*-

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "3"

#
# Description :
# Patching training data for QSMnet	
#
# Input : 
#  RDF : the tissue field
#  QSM : the COSMOS susceptibility 
#  x : Serial number of subjects
#  y : the orientation of each subject
#  z : the number for data augmentation
#
# Outputs:
#  p_RDF : the patches of tissue field
#  p_QSM : the patches of COSMOS susceptibility
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

# sess = tf.Session()
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

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
# K.set_session(sess)
import numpy as np
import random
import matplotlib.pyplot as plt
#from scipy.misc import imread, imresize
import scipy.misc
from scipy.stats import norm
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import load_model
import math
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard  
from IPython.display import clear_output
from model.customized_layer import MirrorPadding3D
from model.common import generate_multi_model
from keras.utils import np_utils

VOXEL_SIZE = (1, 1, 1)
PATCH_SIZE = (64, 64, 64)
EXTRACTION_STEP = (21, 21, 21)
patches_RDFs = []
patches_MASKs = []
patches_QSMs = []

x = [1,2,3]        
y = [1,2,3,4,5]        
z = [1,2,3]  
global i_Total        
i_Total = 0  
for i_x in x:
    for i_y in y:
        for i_z in z:
            i_case = i_x
            FD_DATA = '/public/siwenbin/DLQSM_YH/Aug_test/Demo12/data/traindata/'
            filename = '{0}/{1}/RDF.mat'.format(FD_DATA, i_case)
            RDF = np.real(load_mat(filename, varname='RDF'))
            filename = '{0}/{1}/COSMOS.mat'.format(FD_DATA, i_case)
            QSM = np.real(load_mat(filename, varname='COSMOS'))
                   
            i_dir = i_y-1
            rdf = RDF[...,i_dir]
            qsm = QSM[...,i_dir]
            
            RDF_aug = augment_data(rdf, voxel_size = VOXEL_SIZE, flip='', thetas=[-15, 15])
            QSM_aug = augment_data(qsm, voxel_size = VOXEL_SIZE, flip='', thetas=[-15, 15]) 
            del QSM,RDF
               
            i_aug = i_z-1
            patches_RDF = extract_patches(RDF_aug[i_aug, ...], PATCH_SIZE, EXTRACTION_STEP)
            patches_MASK = abs(patches_RDF) > 1e-2
            patches_MASK.astype(float)
            patches_QSM = extract_patches(QSM_aug[i_aug, ...], PATCH_SIZE, EXTRACTION_STEP)
            # filter out background patch
            idxs_valid = patches_MASK.mean(axis=(1,2,3)) > 0.1      
                     
            patches_RDF = patches_RDF[idxs_valid, ...]
            patches_MASK = patches_MASK[idxs_valid, ...]
            patches_QSM = patches_QSM[idxs_valid, ...]
            
            i_number = patches_RDF.shape[0]
            
            i_Total_pre = i_Total
            i_Total = i_Total + i_number
               
            CASES_TRAIN = np.linspace(i_Total_pre,i_Total,num=i_number, endpoint=False,dtype=np.int)   
            
            for i_case in CASES_TRAIN:
                filename = '/public/siwenbin/DLQSM_YH/Aug_test/Demo12/data/patch/train/p'+str(i_case)+'.h5'
                f = h5py.File(filename,'w') 
                f['p_RDF'] = patches_RDF[(i_case-i_Total_pre),...] 
                f['p_QSM'] = patches_QSM[(i_case-i_Total_pre),...]
                f['i_shape'] = i_Total
                f.close()



                
               
