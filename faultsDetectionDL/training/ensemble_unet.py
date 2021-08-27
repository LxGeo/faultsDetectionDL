# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 00:06:29 2021

@author: cherif
"""

import tensorflow as tf
import segmentation_models as sm
import glob
from skimage.io import imread
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras 
from tensorflow.keras import backend as K

from keras.utils import multi_gpu_model

from keras.utils import normalize
from keras.metrics import MeanIoU
from tqdm import tqdm
from faultsDetectionDL.utils.image_transformation import recurse_transform, images_transformations_list
from faultsDetectionDL.training.losses import surface_loss_keras, segmentation_boundary_loss, Semantic_loss_functions
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
IMG_CHANNELS=3
n_classes=3 #Number of classes for segmentation
class_weights= [0.95,0.03,0.02]
activation='softmax'

DATA_PATH_TMPL="./data/processed/fnc_partition_256_Site_A_B/{}"
TRAIN_PATH = DATA_PATH_TMPL.format("train")
path, dirs, train_ids = next(os.walk(TRAIN_PATH + '/image/'))
VALID_PATH = DATA_PATH_TMPL.format("valid")
valid_path, dirs, valid_ids = next(os.walk(VALID_PATH + '/image/'))
models_path="./models/ensemble/jacc_n_catce"
if (not os.path.isdir(models_path)):
    os.makedirs(models_path)


rt = recurse_transform((np.zeros((5,5)), np.zeros((5,5))))
rt.run_recurse(rt.image_couple, images_transformations_list, [])
augmentation_factor = len(rt.all_transformed)
augmented_train_cnt = augmentation_factor*len(train_ids)
augmented_valid_cnt = augmentation_factor*len(valid_ids)

train_images = np.empty((augmented_train_cnt,SIZE_X, SIZE_Y,IMG_CHANNELS), dtype=np.int8) 
train_masks =np.empty((augmented_train_cnt,SIZE_X, SIZE_Y,1), dtype=np.byte) 

train_im_count=0
print('Reading the training RGB Images and masks:\n')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    c_img = imread(path + id_)[:,:,:IMG_CHANNELS]  
    if(c_img.max()<2):
        continue
    
    c_mask = imread(TRAIN_PATH + '/gt/' + id_).astype(int)#[:,:,1]
    
    #mask = np.expand_dims(mask, axis=-1)
    RT = recurse_transform((c_img, c_mask))
    RT.run_recurse(RT.image_couple, images_transformations_list, [])
    c_all_trans = RT.all_transformed
    for aug_n ,c_augmented_couple in enumerate(c_all_trans) :
        #train_images.append(c_augmented_couple[0]*255)
        #train_masks.append( np.expand_dims(c_augmented_couple[1], axis=-1) )
        train_images[train_im_count, :,:,:] = (c_augmented_couple[0]).astype(np.int8)
        train_masks[train_im_count, :,:,:] = np.expand_dims(c_augmented_couple[1], axis=-1)
        train_im_count=train_im_count+1

train_images = train_images[:train_im_count,:,:,:]
train_masks = train_masks[:train_im_count,:,:,:]
#train_images = np.array(train_images)
#train_masks = np.array(train_masks)


valid_images = np.empty((augmented_valid_cnt,SIZE_X, SIZE_Y,IMG_CHANNELS), dtype=np.int8) 
valid_masks = np.empty((augmented_valid_cnt,SIZE_X, SIZE_Y,1), dtype=np.byte) 

valid_im_count=0
print('Reading the valid RGB Images:\n')
for n, id_ in tqdm(enumerate(valid_ids), total=len(valid_ids)):
    img = imread(valid_path + id_)[:,:,:IMG_CHANNELS]
    mask_valid = imread(VALID_PATH + '/gt/' + id_).astype(int)
    RT = recurse_transform((img, mask_valid))
    RT.run_recurse(RT.image_couple, images_transformations_list, [])
    c_all_trans = RT.all_transformed
    for aug_n ,c_augmented_couple in enumerate(c_all_trans):
        valid_images[valid_im_count, :,:,:] = (c_augmented_couple[0]).astype(np.int8)
        valid_masks[valid_im_count, :,:,:] = np.expand_dims(c_augmented_couple[1], axis=-1)
        valid_im_count+=1
    
    #valid_images.append( img*255)
    #mask_valid = np.expand_dims(mask_valid, axis=-1)
    #valid_masks.append( mask_valid)

#valid_images = np.array(valid_images)
#valid_masks = np.array(valid_masks)

############################################# Save load #######################

train_np_save_path = os.path.join(TRAIN_PATH, "train.npz")
#np.savez(train_np_save_path, train_images, train_masks)
loaded_train = np.load(train_np_save_path)
train_images, train_masks = [loaded_train[k] for k in loaded_train ]

valid_np_save_path = os.path.join(VALID_PATH, "valid.npz")
#np.savez(valid_np_save_path, valid_images, valid_masks)
loaded_valid = np.load(valid_np_save_path)
valid_images, valid_masks = [loaded_valid[k] for k in loaded_valid ]

############################## CAtegorical enconding ########################

from keras.utils import to_categorical
train_masks_cat = to_categorical(train_masks, dtype="uint8")
#y_train_cat = train_masks_cat.reshape((train_masks.shape[0], train_masks.shape[1], train_masks.shape[2], n_classes))
del train_masks


valid_masks_cat = to_categorical(valid_masks, dtype="uint8")
#y_valid_cat = valid_masks_cat.reshape((valid_masks.shape[0], valid_masks.shape[1], valid_masks.shape[2], n_classes))
del valid_masks
###############################




def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        # transform back to logits
        _epsilon = K.epsilon()
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))
        
    return tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=95.98)


def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array(class_weights)) 
focal_loss = sm.losses.CategoricalFocalLoss(alpha=0.9)
categorical_crossentropy_loss = sm.losses.CategoricalCELoss(class_weights=class_weights)
jaccard_loss = sm.losses.JaccardLoss(class_weights=class_weights)
total_loss = categorical_crossentropy_loss + (1*jaccard_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

################################################## callbakcs ##############

def get_model_callbacks(model_save_path):
    """
    """
    #earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    #mcp_save_best = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')
    mcp_save_all = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save_all]
    


########################################################################
###Model 1
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = train_images#preprocess_input1(train_images.copy())
X_test1 = valid_images#preprocess_input1(valid_images.copy())

# define model
model1 = sm.Unet(BACKBONE1 , encoder_weights='imagenet', classes=n_classes, activation=activation)

#model1 = multi_gpu_model(model1, gpus=2)
# compile keras model with defined optimozer, loss and metrics
model1.compile(optim, loss=total_loss, metrics=metrics)

#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model1.summary())


history1=model1.fit(X_train1, 
          train_masks_cat,
          batch_size=32, 
          epochs=50,
          verbose=1,
          validation_data=(X_test1, valid_masks_cat),
          callbacks = get_model_callbacks(models_path+'/res34_backbone_epoch_{epoch:02d}.hdf5'))


#model1.save(models_path+'/res34_backbone_50epochs.hdf5')

############################################################
###Model 2

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(train_images)
X_test2 = preprocess_input2(valid_images)

# define model
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=n_classes, activation=activation)


# compile keras model with defined optimozer, loss and metrics
model2.compile(optim, total_loss, metrics)
#model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model2.summary())


history2=model2.fit(X_train2, 
          train_masks_cat,
          batch_size=16, 
          epochs=50,
          verbose=1,
          validation_data=(X_test2, valid_masks_cat),
          callbacks = get_model_callbacks(models_path+'/inceptionv3_backbone_epoch_{epoch:02d}.hdf5'))


#model2.save(models_path+'/inceptionv3_backbone_50epochs.hdf5')

#####################################################
###Model 3

BACKBONE3 = 'vgg16'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# preprocess input
X_train3 = preprocess_input3(train_images)
X_test3 = preprocess_input3(valid_images)


# define model
model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model3.compile(optim, total_loss, metrics)
#model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model3.summary())

history3=model3.fit(X_train3, 
          train_masks,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test3, valid_masks),
          callbacks = get_model_callbacks(models_path+'/vgg19_backbone_early_epoch.hdf5'))


#model3.save(models_path+'/vgg19_backbone_50epochs.hdf5')

##########################################################
#### Model 4
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(train_images)
X_test1 = preprocess_input1(valid_images)

# define model
model4 = sm.Linknet(BACKBONE1 , encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model4.compile(optim, total_loss, metrics=metrics)

#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model4.summary())


history1=model4.fit(X_train1, 
          train_masks,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test1, valid_masks),
          callbacks = get_model_callbacks(models_path+'/res34_backbone_linknet_early_epoch.hdf5'))

##########################################################
#### Model 5
BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(train_images)
X_test2 = preprocess_input2(valid_images)

# define model
model5 = sm.Linknet(BACKBONE2 , encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model5.compile(optim, total_loss, metrics=metrics)

#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model5.summary())


history1=model5.fit(X_train2, 
          train_masks,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test2, valid_masks),
          callbacks = get_model_callbacks(models_path+'/inceptionv3_backbone_linknet_epoch_{epoch:02d}.hdf5'))

##########################################################

###
#plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#####################################################

from keras.models import load_model

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('saved_models/res34_backbone_50epochs.hdf5', compile=False)
model2 = load_model('saved_models/inceptionv3_backbone_50epochs.hdf5', compile=False)
model3 = load_model('saved_models/vgg19_backbone_50epochs.hdf5', compile=False)

#Weighted average ensemble
models = [model1, model2, model3]
#preds = [model.predict(X_test) for model in models]

pred1 = model1.predict(X_test1)
pred2 = model2.predict(X_test2)
pred3 = model3.predict(X_test3)

preds=np.array([pred1, pred2, pred3])

#preds=np.array(preds)
weights = [0.3, 0.5, 0.2]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

y_pred1_argmax=np.argmax(pred1, axis=3)
y_pred2_argmax=np.argmax(pred2, axis=3)
y_pred3_argmax=np.argmax(pred3, axis=3)


#Using built in keras function
n_classes = 4
IOU1 = MeanIoU(num_classes=n_classes)  
IOU2 = MeanIoU(num_classes=n_classes)  
IOU3 = MeanIoU(num_classes=n_classes)  
IOU_weighted = MeanIoU(num_classes=n_classes)  

IOU1.update_state(y_test[:,:,:,0], y_pred1_argmax)
IOU2.update_state(y_test[:,:,:,0], y_pred2_argmax)
IOU3.update_state(y_test[:,:,:,0], y_pred3_argmax)
IOU_weighted.update_state(y_test[:,:,:,0], weighted_ensemble_prediction)


print('IOU Score for model1 = ', IOU1.result().numpy())
print('IOU Score for model2 = ', IOU2.result().numpy())
print('IOU Score for model3 = ', IOU3.result().numpy())
print('IOU Score for weighted average ensemble = ', IOU_weighted.result().numpy())
###########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy

import pandas as pd
df = pd.DataFrame([])

for w1 in range(0, 4):
    for w2 in range(0,4):
        for w3 in range(0,4):
            wts = [w1/10.,w2/10.,w3/10.]
            
            IOU_wted = MeanIoU(num_classes=n_classes) 
            wted_preds = np.tensordot(preds, wts, axes=((0),(0)))
            wted_ensemble_pred = np.argmax(wted_preds, axis=3)
            IOU_wted.update_state(y_test[:,:,:,0], wted_ensemble_pred)
            print("Now predciting for weights :", w1/10., w2/10., w3/10., " : IOU = ", IOU_wted.result().numpy())
            df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                         'wt3':wts[2], 'IOU': IOU_wted.result().numpy()}, index=[0]), ignore_index=True)
            
max_iou_row = df.iloc[df['IOU'].idxmax()]
print("Max IOU of ", max_iou_row[3], " obained with w1=", max_iou_row[0],
      " w2=", max_iou_row[1], " and w3=", max_iou_row[2])         


#############################################################
opt_weights = [max_iou_row[0], max_iou_row[1], max_iou_row[2]]

#Use tensordot to sum the products of all elements over specified axes.
opt_weighted_preds = np.tensordot(preds, opt_weights, axes=((0),(0)))
opt_weighted_ensemble_prediction = np.argmax(opt_weighted_preds, axis=3)
#######################################################
#Predict on a few images

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,:]
test_img_input=np.expand_dims(test_img_norm, 0)

#Weighted average ensemble
models = [model1, model2, model3]

test_img_input1 = preprocess_input1(test_img_input)
test_img_input2 = preprocess_input2(test_img_input)
test_img_input3 = preprocess_input3(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_pred2 = model2.predict(test_img_input2)
test_pred3 = model3.predict(test_img_input3)

test_preds=np.array([test_pred1, test_pred2, test_pred3])

#Use tensordot to sum the products of all elements over specified axes.
weighted_test_preds = np.tensordot(test_preds, opt_weights, axes=((0),(0)))
weighted_ensemble_test_prediction = np.argmax(weighted_test_preds, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(weighted_ensemble_test_prediction, cmap='jet')
plt.show()

#####################################################################