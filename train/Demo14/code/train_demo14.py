# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#
# Description:
#  Training code of DIAM-CNN
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

# sess = tf.Session()
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

import sys
sys.path.append("/public/siwenbin/DLQSM_YH/")
sys.path.append("/public/siwenbin/DLQSM_YH/model/")

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
import glob 
import re
from customized_layer import MirrorPadding3D
from keras.utils import CustomObjectScope

i_Train = 0
i_Valid = 0
def generator_js(pd):
    global i_Train
    global i_Valid
    while 1:
        patches_RDFs = []
        patches_MASKs = []
        patches_QSMs = []
        patches_RDFs1 = []
        patches_RDFs2 = []
        if pd == 1:
            for i_case in range(int(batch_size)):
                filepath = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/patch/train/RDF/p'+str(i_Train)+'.h5'
                filepath1 = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/patch/train/RDF1/p'+str(i_Train)+'.h5'
                filepath2 = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/patch/train/RDF2/p'+str(i_Train)+'.h5'
                i_Train = i_Train + 1
                print('patch number:    ',i_Train)
                
                f = h5py.File(filepath,'r')
                f.keys()                           
                patch_RDF = f['p_RDF'][:]            
                patch_QSM = f['p_QSM'][:] 
                f.close()
                
                f1 = h5py.File(filepath1,'r')
                f1.keys()                         
                patch_RDF1 = f1['p_RDF'][:]            
                f1.close()
                
                f2 = h5py.File(filepath2,'r')
                f2.keys()                       
                patch_RDF2 = f2['p_RDF'][:]            
                f2.close()
                
                patches_RDFs.append(patch_RDF[np.newaxis, ...])
                patches_QSMs.append(patch_QSM[np.newaxis, ...])
                patches_RDFs1.append(patch_RDF1[np.newaxis, ...])
                patches_RDFs2.append(patch_RDF2[np.newaxis, ...])
                
                if i_Train>=a:
                    i_Train = 0
                    
            in_x = np.concatenate(patches_RDFs, axis=0)[..., np.newaxis]
            in_x1 = np.concatenate(patches_RDFs1, axis=0)[..., np.newaxis]
            in_x2 = np.concatenate(patches_RDFs2, axis=0)[..., np.newaxis]
            
            IN_X = np.concatenate((in_x,in_x1,in_x2),axis=-1);
            in_y = np.concatenate(patches_QSMs, axis=0)[..., np.newaxis]
                
        elif pd ==2:
             for i_case in range(int(batch_size)):
                 filepath = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/patch/valid/RDF/p'+str(i_Valid)+'.h5'
                 filepath1 = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/patch/valid/RDF1/p'+str(i_Valid)+'.h5'
                 filepath2 = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/data/patch/valid/RDF2/p'+str(i_Valid)+'.h5'
                 i_Valid = i_Valid + 1
                 f = h5py.File(filepath,'r')
                 f.keys()                      
                 patch_RDF = f['p_RDF'][:]            
                 patch_QSM = f['p_QSM'][:] 
                 f.close()
                 
                 f1 = h5py.File(filepath1,'r')
                 f1.keys()                           
                 patch_RDF1 = f1['p_RDF'][:]            
                 f1.close()
                
                 f2 = h5py.File(filepath2,'r')
                 f2.keys()                         
                 patch_RDF2 = f2['p_RDF'][:]            
                 f2.close()
                 
                 patches_RDFs.append(patch_RDF[np.newaxis, ...])
                 patches_QSMs.append(patch_QSM[np.newaxis, ...])
                 patches_RDFs1.append(patch_RDF1[np.newaxis, ...])
                 patches_RDFs2.append(patch_RDF2[np.newaxis, ...])
                
                 if i_Valid>=b:
                     i_Valid = 0
             in_x = np.concatenate(patches_RDFs, axis=0)[...,np.newaxis]
             in_x1 = np.concatenate(patches_RDFs1, axis=0)[..., np.newaxis]
             in_x2 = np.concatenate(patches_RDFs2, axis=0)[..., np.newaxis]
             IN_X = np.concatenate((in_x,in_x1,in_x2),axis=-1);
             in_y = np.concatenate(patches_QSMs, axis=0)[...,np.newaxis]
                
        yield IN_X,in_y       

def findLastCheckpoint(save_dir):
    """
    Find the last saved model number.
    ### ARGUMENTS
    - save_dir, dir of the saved model.

    ### RETURN
    - initial_epoch, the number of last saved model.
    """
    file_list = glob.glob(os.path.join(save_dir,'model_*.h5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch
    
if __name__ == '__main__':
    VOXEL_SIZE = (1, 1, 1)
    PATCH_SIZE = (64, 64, 64)
    EXTRACTION_STEP = (21, 21, 21)
    batch_size = 8   
    three_loss = generate_loss1
     
    model, model_single = generate_multi_model(1, PATCH_SIZE, use_bn = True, multi_input = True) 
    model.compile(loss= three_loss, optimizer = Adam(lr = 1e-3), metrics=[l1_loss, gradient_loss, dipole_loss])

    EPOCH = 160
    model_dir = '../tmp_model/js'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=False, period=1)
    
    DIR_WEIGHT ='/public/siwenbin/DLQSM_YH/Aug_test/Demo14/tmp_model/'
    checkpoint = ModelCheckpointMultiGPU(model_real=model_single,                                                                                          
                                         filepath='{0}/net_tmp1.h5'.format(DIR_WEIGHT),
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True)
    
    DIR_LOG = '/public/siwenbin/DLQSM_YH/Aug_test/Demo14/log/test_log'
    tensorboard = TensorBoard(log_dir=DIR_LOG,
                              histogram_freq=0,
                              batch_size=batch_size,
                              write_graph=False,
                              write_grads=True,
                              write_images=False)
    
    reduceLR = ReduceLROnPlateauBestWeight(model_load=model_single,
                                         filepath='{0}/net_tmp1.h5'.format(DIR_WEIGHT),
                                         monitor='val_loss',
                                         factor=0.1,
                                         patience=20,
                                         verbose=0, 
                                         mode='auto', 
                                         cooldown=0, 
                                         min_lr=1e-6)
    
    callbacks = [checkpointer,checkpoint,tensorboard,reduceLR] 
    
    initial_epoch = findLastCheckpoint(save_dir=model_dir)
    if initial_epoch > 0:
        print('Resuming by loading epoch %03d'%initial_epoch)
        #model.load_weight(os.path.join(model_dir,'model_%03d.h5'%initial_epoch))#,custom_objects={'':})
        with CustomObjectScope({'MirrorPadding3D':MirrorPadding3D,'generate_loss1':generate_loss1,'l1_loss':l1_loss,'gradient_loss':gradient_loss,'dipole_loss':dipole_loss}):
            model = load_model(os.path.join(model_dir,'model_%03d.h5'%initial_epoch))
#        model = load_model(os.path.join(model_dir,'model_%03d.h5'%initial_epoch),custom_objects={'MirrorPadding3D':MirrorPadding3D,'three_loss':three_loss})
    
    model.fit_generator(generator_js(1), 
            steps_per_epoch=math.floor(a/batch_size), 
            epochs=EPOCH,verbose=1, 
            callbacks=callbacks, 
            validation_data=generator_js(2), 
            validation_steps=int(math.floor(b/batch_size)),  
            max_queue_size=10, 
            workers=1,     
            use_multiprocessing=False, 
            shuffle=True, 
            initial_epoch=initial_epoch)  
    
    model.save_weights('/public/siwenbin/DLQSM_YH/Aug_test/Demo14/Model/Model_DIAMCNN_Demo14.h5')






