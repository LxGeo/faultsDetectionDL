#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:59:19 2021

@author: cherif
"""

import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from faultsDetectionDL.training.pt import TRAIN_PATH,VALID_PATH, IMG_CHANNELS, BATCH_SIZE, \
    n_classes, class_weights, activation, models_path, model_name_template
    

from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity
from torch.utils.data import TensorDataset, DataLoader
from faultsDetectionDL.data.faults_dataset import FaultsDataset 

from matplotlib import pyplot as plt



############### Load data
#valid_dataset = FaultsDataset(VALID_PATH, [Trans_Identity()])
#valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


#################### Load model
from faultsDetectionDL.training.pt.pt_segmentation_models import model, valid_dataloader, c_models_path

model_idx = 11
c_model_path = os.path.join(c_models_path,'{}_model.pth'.format(model_idx))

model=torch.load(c_model_path)

img, gt = list(iter(valid_dataloader))[2]
img = img.float().cuda()
y = model.forward(img)
mask_pred = y.cpu().permute(0,2,3,1).detach().numpy()

image_index = 9
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(img[image_index, :,:,: ].permute(1,2,0).cpu().numpy().astype(int), cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(gt[image_index, :,:,: ].permute(1,2,0).cpu(), cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(mask_pred[image_index], cmap='jet')
plt.show()