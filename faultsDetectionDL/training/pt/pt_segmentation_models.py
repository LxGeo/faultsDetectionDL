#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:05:26 2021

@author: cherif
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp
from faultsDetectionDL.training.pt import TRAIN_PATH,VALID_PATH, IMG_CHANNELS, BATCH_SIZE, \
    n_classes, class_weights, activation, models_path, model_name_template, DEVICE
from torch.utils.data import TensorDataset, DataLoader
from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity
from faultsDetectionDL.data.pt.faults_dataset import FaultsDataset 
import json
from json.decoder import JSONDecodeError

################# Model1
model_name = "MAnet_resnext101_32x8d"
ENCODER = 'resnext101_32x8d'
ENCODER_WEIGHTS = 'imagenet'
model = smp.MAnet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS, 
    in_channels=IMG_CHANNELS,
    classes=n_classes,
    activation=activation,
    decoder_use_batchnorm=True,
    decoder_channels=[512,512,256,128,64],
    #decoder_attention_type="scse"
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


############### Load data
train_dataset = FaultsDataset(TRAIN_PATH, images_transformations_list, preprocessing=preprocessing_fn, img_bands=IMG_CHANNELS, num_classes=n_classes)
valid_dataset = FaultsDataset(VALID_PATH, [Trans_Identity()], preprocessing=preprocessing_fn, img_bands=IMG_CHANNELS, num_classes=n_classes)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


################# Optimizer definition
metrics = [
    #smp.utils.metrics.IoU(threshold=0.5),
    #smp.utils.metrics.Fscore()
]

LR = 0.000291
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=LR),
])


################# loss definition
from faultsDetectionDL.training.pt.custom_losses import wrap_jaccard_loss,wrap_ce_loss
total_loss = wrap_ce_loss(torch.tensor(class_weights).to(DEVICE)) + 0.2*wrap_jaccard_loss() 
loss_name = "weighted_cc_n_jacc"


####################

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=total_loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=total_loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


c_model_name = model_name_template.format(model_name=model_name, loss_name=loss_name)
c_models_path = os.path.join(models_path, c_model_name)
if (not os.path.isdir(c_models_path)):
    os.makedirs(c_models_path)

logs_file_path=os.path.join(c_models_path, "logs.json")
open(logs_file_path, "a").close()
if __name__ =="__main__":
    
    all_logs={}
    for i in range(1, 10):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
                
        # do something (save model, change lr, etc.)
        torch.save(model, os.path.join(c_models_path,'{}_model.pth'.format(i)))
        torch.cuda.empty_cache()
        print('Model saved!')
        
        ## load json file
        old_logs=dict()
        try:
            with open(logs_file_path, "r") as logs_file:
                old_logs = json.load(logs_file)
        except JSONDecodeError:
            pass
        
        old_logs[i]={"train_logs":train_logs, "valid_logs":valid_logs}
        
        with open(logs_file_path, "w") as logs_file:
            json.dump(old_logs, logs_file)
        
        
        if i % 4==0:
            optimizer.param_groups[0]['lr'] /= 2
            print('Decrease decoder learning rate by factor 2!')
    
    
