import tensorflow as tf
import keras.backend as K
from .UNet_3d import UNet_3d
from .UNet_3d555 import UNet_3d555
from .UNet_3d2 import UNet_3d2
from .UNet_3d1 import UNet_3d1
from .UNet_3d_sigmoid import UNet_3d_sigmoid
from .customized_layer import DipoleConv
from keras.models import Model

#
#create model
#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

#   Configure TensorFlow session (memory allocation)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def generate_model(*args, model_name='rdf_ind',
                          aug_name='',
                          param_aug={},
                          **kwargs):
    
    # select basic model (RDF --> QSM)
    #if model_name == 'UNet3d':
    func_generate_single = UNet_3d
    # create basic model
    model_basic = func_generate_single(*args, **kwargs)
    
    # create aug model
    if aug_name == 'rdf_ind':
        # induced field (QSM --> RDF)
        rdf_ind = DipoleConv(voxel_size=param_aug.get('voxel_size', (1,1,1)), 
                             B0_dir=param_aug.get('B0_dir', (0,0,1)),
                             name=aug_name)(model_basic.output)
        # combine models (RDF --> [QSM, RDF])
        model = Model(inputs=model_basic.inputs, outputs=[model_basic.output, rdf_ind])
        
    else:
        model = model_basic
    
    return model


def generate_multi_model(n_GPU, *args, **kwargs): 
    from keras.utils import multi_gpu_model
    print('Using {0} GPU'.format(n_GPU))
    if n_GPU <= 1:
        model_single = generate_model(*args, **kwargs)
        model = model_single
    else:
        with tf.device("/cpu:0"):
        # initialize the model on CPU
            model_single = generate_model(*args, **kwargs)
        model = multi_gpu_model(model_single, gpus=n_GPU)
    return model, model_single
