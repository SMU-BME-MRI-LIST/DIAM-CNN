import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
import h5py

#
# Copyright @ Yanqiu Feng
# Laboratory for Medical Imaging and Diagnostic Technology
# Southern Medical University
# email: foree@163.com
#

class ModelCheckpointMultiGPU(ModelCheckpoint):
    '''
    Fix bugs that in ModelCheckpoint, the multi_gpu_model rather than
    the real model is saved
    '''

    def __init__(self, model_real, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Save the real model
        self.model_real = model_real
        

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    print('Can save best model only with %s available, skipping.' % (self.monitor))
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_real.save_weights(filepath, overwrite=True)
                        else:
                            self.model_real.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_real.save_weights(filepath, overwrite=True)
                else:
                    self.model_real.save(filepath, overwrite=True) 
#                    with h5py.File(filepath,'a') as f:
#                     f['p_epoch'] = range(epoch)
#                     f['p_epoch1'] = epoch
#                    f.close() 
                    
class TensorBoardImage(TensorBoard):
    '''
    TensorBoard with image plotting per epoch
    '''

    def __init__(self, flag_plot=False, 
                       idxs=[0],
                       idx_slice=0,
                       *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.flag_plot = flag_plot
        self.idxs = idxs
        self.idx_slice = idx_slice
        self.summary_op_plot = None
        

    def set_model(self, model):
        super().set_model(model)
        
        if self.flag_plot:
            # plot output image
            img_ip = self.model.inputs[0]
            tf.summary.image("input", img_ip[..., self.idx_slice, 0:1], max_outputs=len(self.idxs))
            img_pred = self.model.outputs[0]
            tf.summary.image("pred", img_pred[..., self.idx_slice, 0:1], max_outputs=len(self.idxs))
            img_gt = self.model.targets[0]
            tf.summary.image("truth", img_gt[..., self.idx_slice, 0:1], max_outputs=len(self.idxs))
            self.summary_op_plot = tf.summary.merge_all()
            
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.flag_plot:
            if epoch % 1 == 0:

                data = self.validation_data
                
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(data) == len(tensors)
                
                # show specific slices
                if self.model.uses_learning_phase:
                    batch_val = [x[self.idxs] for x in data[:-1]]
                    batch_val.append(data[-1])
                else:
                    batch_val = [x[self.idxs] for x in data]
                assert len(batch_val) == len(tensors)
                feed_dict = dict(zip(tensors, batch_val))
                result = self.sess.run([self.summary_op_plot], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)   
                    
        super().on_epoch_end(epoch=epoch, logs=logs)  
        
        
        
class ReduceLROnPlateauBestWeight(ReduceLROnPlateau):
    '''
    Reduce learning rate when a metric has stopped improving.
    When LR changes, load the best weight
    '''

    def __init__(self, model_load, filepath, 
                       *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_load = model_load
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        # load best weight
                        self.model_load.load_weights(self.filepath)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1          
                
                

class Snapshot(Callback):
    '''
    Snapshot the best weights at given epochs
    '''
 
    def __init__(self, epoch_snap, filepath, format_snap, 
                       *args, **kwargs): 
 
        super().__init__(*args, **kwargs) 
         
        self.epoch_snap = list(epoch_snap) 
        self.filepath = filepath 
        self.format_snap = format_snap 
         
 
    def on_epoch_end(self, epoch, logs=None): 
        from shutil import copyfile 
         
        if epoch in self.epoch_snap: 
            copyfile(self.filepath, self.format_snap.format(epoch)) 
            
            
            
# Define callbacks for generator handling
class HandleGen(Callback):
    def __init__(self, gen, gen_val,
                       *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.gen = gen
        self.gen_val = gen_val
 
    def on_epoch_begin(self, epoch, logs=None):
        self.gen.reset()
        self.gen_val.reset()
        
    def on_train_end(self, logs=None):
        self.gen.stop()
        self.gen_val.stop()