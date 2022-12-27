# -*- coding: utf-8 -*-
"""
@author: Kevin Freire

This network archtecture is designed using Fucntional API.  
"""
import tensorflow as tf
import argparse
#Model Components
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Add, concatenate, Input, subtract

# Spatial Attention
def SA(feat_map):
    conv1 = Conv2D(64, (1,1), strides=1, padding='valid')(feat_map)
    conv2 = Conv2D(64, (1,1), strides=1, padding='valid')(feat_map)
    conv3 = Conv2D(64, (3,3), strides=1, activation='relu', padding='same')(feat_map)
    conv3 = Conv2D(64, (3,3), strides=1, padding='same')(conv3)
    
    conv4 = Multiply()([conv1, conv2])
    conv4_softmax = tf.keras.activations.softmax(conv4)
    conv5 = Multiply()([conv3, conv4_softmax])
    conv5 = Conv2D(64, (1,1), strides=1, padding='valid')(conv5)
    
    sa_feat_map = Add()([feat_map, conv5])
    
    return sa_feat_map

# Channel Attention 
def CA(feat_map):

    conv1 = AveragePooling2D(pool_size=(1,1), strides=1, padding='valid')(feat_map)
    conv2 = Conv2D(64, (1,1), strides=1, activation='relu', padding='valid')(conv1)  
    conv3 = Conv2D(64, (1,1), strides=1, activation='sigmoid',padding='valid')(conv2)
    
    ca_feat_map = Multiply()([feat_map, conv3])
    
    return ca_feat_map

# BAFB -- four for each BMG in the denoiser
def BAFB(input_bafb):
    fcr1 = Conv2D(64, (1,1), strides=1, activation='relu', padding='valid')(input_bafb)
    
    fsa1 = SA(fcr1)
    fes_up = Add()([fsa1,fcr1])
    
    fca1 = CA(fcr1)
    fes_down = Add()([fca1, fcr1])
    
    fca2 = CA(fes_up)
    fsa2 = SA(fes_down)
    
    fcr2=concatenate([fca2, fes_up, fes_down, fsa2],axis=3)
    fcr2=Conv2D(1, (1,1), strides=1, activation='relu', padding='valid')(fcr2) 
    
    fc=concatenate([fcr1, fcr2],axis=3)
    fc=Conv2D(1, (1,1), strides=1, padding='valid')(fc) 
    
    return fc

# Boosting Module Groups (BMG)
def BMG(bmg_input):
    bafb1 = BAFB(bmg_input)
    bafb2 = BAFB(bafb1)
    bafb3 = BAFB(bafb2)
    bafbn = BAFB(bafb3)   
    fg = Add()([bmg_input,bafbn]) #group skip connection 
    return fg

def dilated_conv_block(dc_input, num_filters, kernel_size, dilation_rate, padding='same', activation='relu'):
    x = Conv2D(num_filters, kernel_size, dilation_rate=dilation_rate, padding=padding, activation=activation)(dc_input)
    x = Conv2D(num_filters, kernel_size, dilation_rate=dilation_rate, padding=padding, activation=activation)(x)
    return x

def MD_RFA():
    inputs = (None, None, 1)
    inputs=Input(shape=inputs)

    fm1 = dilated_conv_block(inputs, 32, 3, 1)
    fm2 = dilated_conv_block(inputs, 32, 3, 2)
    fm3 = dilated_conv_block(inputs, 32, 3, 4)
    fm4 = concatenate([fm1, fm2, fm3], axis=-1)
    fm_out = Conv2D(64, 3, dilation_rate=2, padding='same', activation='relu')(fm4)

    bmg1= BMG(fm_out)
    bmg2= BMG(bmg1)
    bmg3 = BMG(bmg2)

    s1 = Add()([fm_out, bmg3])

    decoder1 = Conv2D(64, 3, dilation_rate=2, padding='same', activation='relu')(s1)
    decoder2 = Conv2D(64, 3, dilation_rate=2, padding='same', activation='relu')(decoder1)
    decoder3 = Conv2D(3, 3, dilation_rate=2, padding='same')(decoder2)

    out = subtract([inputs, decoder3])

    model = Model(inputs=[inputs], outputs=[out, out, out])
    return model

if __name__ == '__main__':
    # -------------------------------End---------------------------------------------
    model = MD_RFA()
    model.summary()