#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed SJuly 21 11:36:52 2021

U-Net Model (Tasar et al. 2019)

Build the U-Net model proposed by Tasar et al. in the paper "Incremental Learning for Semantic Segmentation of Large-Scale Remote Sensing Data"

Authors: Onur Tasar, Student member, IEEE, Yuliya Tarabalka, Senior member, IEEE, Pierre Alliez

Journal Reference: IEEE JOURNAL OF SELECTED TOPICS IN APPLIED EARTH OBSERVATIONS AND REMOTE SENSING, 12, 2019, 3524-3537

DOI: 10.1109/JSTARS.2019.2925416
    
@author: Bilel Kanoun
"""

import tensorflow as tf
import keras
#from keras.utils import multi_gpu_model
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout  
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import os,sys
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from skimage.io import imread, imshow
#from skimage.transform import resize 
import scipy.io as sio

from keras import backend as K
import datetime

sys.path.append("../../faultsDetectionDL/")
from utils.image_transformation import recurse_transform, images_transformations_list
    
def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        # transform back to logits
        _epsilon = K.epsilon()
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))
        
    return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=95.98)


def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)


seed = 42
np.random.seed = seed

IMG_HEIGTH=512
IMG_WIDTH=512
IMG_CHANNELS=4

# Read the Data (Sites A and B basic mapping)
DATA_PATH_TMPL="../../data/processed/partition_512_site_A_B/{}/"
TRAIN_PATH = DATA_PATH_TMPL.format("train")

path, dirs, train_ids = next(os.walk(TRAIN_PATH + '/image/'))

VALID_PATH = DATA_PATH_TMPL.format("valid")

valid_path, dirs, valid_ids = next(os.walk(VALID_PATH + '/image/'))


MODELS_PATH = "../../models/{}/".format(datetime.datetime.now())
filepath = MODELS_PATH+'checkpoints_Epochs15_MC_MRef/MC_MRef_epoch_{epoch:02d}.hdf5'

#X_train = np.zeros((len(train_ids), IMG_HEIGTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

#Y_train = np.zeros((len(train_ids), IMG_HEIGTH, IMG_WIDTH, 1), dtype=np.float32)



#%%

# Change Contrast
#X_contrast = tf.image.random_contrast(X_train, lower=0.75, upper=1.25)

# Change brighness
#X_brighness = tf.image.random_brightness(X_train, max_delta=0.8)
#X_brighness = tf.clip_by_value(X_brighness, clip_value_min=0.0, clip_value_max=1.0)


#X_train_ = tf.clip_by_value(X_train_, clip_value_min=0.0, clip_value_max=255.0)
#Y_train_ = tf.clip_by_value(Y_train_, clip_value_min=0.0, clip_value_max=1.0)

                 
#%% Build the model
inputs = Input((IMG_WIDTH,IMG_HEIGTH, IMG_CHANNELS))

mean_list = [123.68, 116.779, 103.939, 161.54] # Mean List of R,G,B and T Bands
s = Lambda(lambda x: x-mean_list)(inputs) # Normalize the patches (substruction to Mean List)

#--------------------- Downsampling Network-----------------------------------
# Block 1
c1 = Conv2D(64, (3,3), activation='relu', padding='same')(s)
c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2,2))(c1) #128x128x4

# Block 2
c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
p2 = MaxPooling2D((2,2))(c2) #64x64x4

# Block 3
c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
p3 = MaxPooling2D((2,2))(c3) #32x32x4

# Block 4
c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
p4 = MaxPooling2D((2,2))(c4) #16x16x4

# Block 5
c5 = Conv2D(512, (3,3), activation='relu', padding='same')(p4)
c5 = Conv2D(512, (3,3), activation='relu', padding='same')(c5)
c5 = Conv2D(512, (3,3), activation='relu', padding='same')(c5)
p5 = MaxPooling2D((2,2))(c5) #8x8x4

# Latent Space (ls)
c_ls = Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same')(p5)

#--------------------- Upsampling Network-----------------------------------
# Block 5'
u5_up = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c_ls)
u5_up = concatenate([u5_up, c5])
c5_up = Conv2D(512, (3,3), activation='relu', padding='same')(u5_up)
c5_up = Conv2D(512, (3,3), activation='relu', padding='same')(c5_up)
c5_up = Conv2D(512, (3,3), activation='relu', padding='same')(c5_up)

# Block 4'
u4_up = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5_up)
u4_up = concatenate([u4_up, c4])
c4_up = Conv2D(512, (3,3), activation='relu', padding='same')(u4_up)
c4_up = Conv2D(512, (3,3), activation='relu', padding='same')(c4_up)
c4_up = Conv2D(512, (3,3), activation='relu', padding='same')(c4_up)

# Block 3'
u3_up = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c4_up)
u3_up = concatenate([u3_up, c3])
c3_up = Conv2D(256, (3,3), activation='relu', padding='same')(u3_up)
c3_up = Conv2D(256, (3,3), activation='relu', padding='same')(c3_up)
c3_up = Conv2D(256, (3,3), activation='relu', padding='same')(c3_up)

# Block 2'
u2_up = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c3_up)
u2_up = concatenate([u2_up, c2])
c2_up = Conv2D(128, (3,3), activation='relu', padding='same')(u2_up)
c2_up = Conv2D(128, (3,3), activation='relu', padding='same')(c2_up)
c2_up = Conv2D(128, (3,3), activation='relu', padding='same')(c2_up)

# Block 1'
u1_up = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c2_up)
u1_up = concatenate([u1_up, c1], axis=3)
c1_up = Conv2D(128, (3,3), activation='relu', padding='same')(u1_up)

# ----------------------Output Generation--------------------------------------
outputs = Conv2D(1, (1,1), activation=tf.nn.sigmoid)(c1_up)

model = Model(inputs=[inputs], outputs=[outputs])

#parallel_model = multi_gpu_model(model, gpus=1)

optimizer_dicc = {'sgd': optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  'rmsprop': optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  'adagrad': optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
                  'adadelta': optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                  'adam': optimizers.Adam(lr=0.0001)}
                   #'adam': optimizers.Adam(lr=0.002, beta_1=0.5, beta_2=0.99, epsilon=1e-08, decay=0.0) # Bahram parameters GAN 


model.compile(optimizer=optimizer_dicc['adam'], loss=weighted_binary_crossentropy) #, metrics=['accuracy'])
#model.compile(optimizer=optimizer_dicc['sgd'], loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#Model CheckPoint - save model every ecpoch
# save best only -> The latest best model will not be overwritten


# Tensor Board Callback

#Earlystop = EarlyStopping(patience=5, monitor='val_loss'),


if __name__ =="__main__":
    
    #### train_data_augmentation
    
    AUG_X_TRAIN = []
    AUG_Y_TRAIN =[]
    
    print('Reading the training RGB Images and masks:\n')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        c_img = imread(path + id_)[:,:,:IMG_CHANNELS]    
        
        c_mask = imread(TRAIN_PATH + '/gt/' + id_)#[:,:,1]
        
        #mask = np.expand_dims(mask, axis=-1)
        RT = recurse_transform((c_img, c_mask))
        RT.run_recurse(RT.image_couple, images_transformations_list, [])
        c_all_trans = RT.all_transformed
        for c_augmented_couple in c_all_trans :
            AUG_X_TRAIN.append(c_augmented_couple[0])
            AUG_Y_TRAIN.append( np.expand_dims(c_augmented_couple[1], axis=-1) )
    
    AUG_X_TRAIN = np.array(AUG_X_TRAIN)
    AUG_Y_TRAIN = np.array(AUG_Y_TRAIN)
        
    
    print("Done!")
    
    
    # Sort the Valid ids
    #valid_ids.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        
    # Valid images
    X_valid = np.zeros((len(valid_ids), IMG_HEIGTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32) 
    Y_valid = np.zeros((len(valid_ids), IMG_HEIGTH, IMG_WIDTH, 1), dtype=np.float32)
    
    
    print('Reading the valid RGB Images:\n')
    for n, id_ in tqdm(enumerate(valid_ids), total=len(valid_ids)):
        img = imread(valid_path + id_)[:,:,:IMG_CHANNELS]
        
        #img = resize(img, (IMG_HEIGTH, IMG_WIDTH), mode='constant', preserve_range=True)  
        X_valid[n] = img
        
        mask_valid = imread(VALID_PATH + '/gt/' + id_)#[:,:,1]
        
        mask_valid = np.expand_dims(mask_valid, axis=-1)
        
        Y_valid[n] = mask_valid
       
    print('Done!')

    if not (os.path.isdir(os.path.dirname(filepath))):
        os.makedirs(os.path.dirname(filepath))
    
    callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', period=1), TensorBoard(log_dir='logs_Tasar_Epochs_15_MC_MRef')] #, save_best_only=True
    
    history = model.fit(AUG_X_TRAIN,AUG_Y_TRAIN, batch_size=12, epochs=15, validation_data=(X_valid, Y_valid), callbacks=[callbacks], verbose=1) #,  validation_split=0.1, steps_per_epoch=1000 
    
    sio.savemat(MODELS_PATH+'/Mean_list_MRef_plus.mat', {"mean_list":mean_list})
    sio.savemat(MODELS_PATH+'/val_loss_MRef_plus.mat', {"val_loss":history.history['val_loss']})
