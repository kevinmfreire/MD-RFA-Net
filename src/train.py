# -*- coding: utf-8 -*-
"""
This code uses perceptual loss, mse and ssim for optimization. also shows the metric psnr and ssim.
@author: Kevin Freire
"""
import tensorflow as tf
import numpy as np
import h5py
import gc

from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from network import MD_RFA
from loss import psnr_loss, dssim_loss, ssim_loss, ResNet50V2Model
from utils import read_hdf5

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
tf.random.set_seed(42)

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def get_all_train_data(dataset_type):

    training_data, training_labels = [], []

    for type_ in dataset_type:
        train_address = '../data/processed/train_32_32_{}_dicom_0.h5'.format(type_)
        data,labels = read_hdf5(train_address)
        ## divison by 4095 keeps the input output between 0-1
        data = (data[:,:,:,None]/4095).astype(np.float32)
        labels = (labels[:,:,:,None]/4095).astype(np.float32)
        labels_3 = np.concatenate((labels,labels,labels),axis=-1)
        training_data.append(data)
        training_labels.append(labels_3)

    train_data = np.concatenate(training_data, axis=0)
    train_labels = np.concatenate(training_labels, axis=0)

    return train_data, train_labels

def train_model(model, data, labels, weight_address, checkpoint_filepath, loss_weight, numOfEpoch = 200, batch_size = 16):

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

    # Loss functions that need initialization
    perceptual = ResNet50V2Model()
    mse = MeanSquaredError()

    loss = [perceptual.perceptual_loss, mse, dssim_loss]
    loss_weights = loss_weight

    ADAM=Adam(learning_rate=0.0002, beta_1=0.01, beta_2=0.999)
    model.compile(optimizer=ADAM,loss=loss, loss_weights=loss_weights,metrics=[psnr_loss, ssim_loss])

    ###Train
    hist_adam = model.fit(x=data,y=[labels,labels,labels],batch_size=batch_size,epochs=numOfEpoch
                        ,validation_split=0, verbose=2, shuffle=True, callbacks=[model_checkpoint_callback, ClearMemory()])
    model.save_weights(weight_address)


if __name__ == '__main__':

    weight_address = 'model/weights/weights_mdrfa_perceptual40_mse30_dsim30.h5'
    checkpoint_filepath = '/model/weights/weights_mdrfa_perceptual40_mse30_dsim30_ckpt.hdf5'
    dataset_type = ['thoracic', 'piglet', 'head_N012', 'chest_C002', 'abdomen_L058']

    model = MD_RFA()
    train_data, train_labels = get_all_train_data(dataset_type)

    train_model(model, train_data, train_labels, weight_address, checkpoint_filepath, loss_weight=[40,30,30])
