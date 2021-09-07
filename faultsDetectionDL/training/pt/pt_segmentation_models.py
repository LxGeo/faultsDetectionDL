#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:05:26 2021

@author: cherif
"""

import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from faultsDetectionDL.training.pt import TRAIN_PATH,VALID_PATH, IMG_CHANNELS, BATCH_SIZE, \
    n_classes, class_weights, activation, models_path, model_name_template
from torch.utils.data import TensorDataset, DataLoader
from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity
from faultsDetectionDL.data.pt.faults_dataset import FaultsDataset 

############### Load data
train_dataset = FaultsDataset(TRAIN_PATH, images_transformations_list)
valid_dataset = FaultsDataset(VALID_PATH, [Trans_Identity()])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


################# Model1
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet", 
    in_channels=IMG_CHANNELS,
    classes=n_classes,
    activation=activation
)

################# Optimizer definition
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

LR = 0.0001
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=LR),
])


################# loss definition
from segmentation_models_pytorch.utils.losses import BCELoss , JaccardLoss
total_loss = BCELoss()+ JaccardLoss()


####################
DEVICE = 'cuda'
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

c_model_name = model_name_template.format(model_name="upp", loss_name="jacc_n_bce")
c_models_path = os.path.join(models_path, c_model_name)
if (not os.path.isdir(c_models_path)):
    os.makedirs(c_models_path)
if __name__ =="__main__":
        
    for i in range(0, 40):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        
        # do something (save model, change lr, etc.)
        torch.save(model, os.path.join(c_models_path,'{}_model.pth'.format(i)))
        torch.cuda.empty_cache()
        print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

