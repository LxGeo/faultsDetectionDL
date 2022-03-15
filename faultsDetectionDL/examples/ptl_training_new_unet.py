#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:42:10 2021

@author: cherif
"""


from faultsDetectionDL.training.ptl.lightning_traing import lightningSegModel
from faultsDetectionDL.data.ptl.faults_data_module import FaultsDataModule
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl

import sys

model_outputs_root_path = sys.argv[1]
tensorboard_logs_path = sys.argv[2]

train_params = {
    "learning_rate":0.01,
    "batch_size":8,
    "DEVICE":"gpu",
    "model_outputs_root_path":model_outputs_root_path,#"./models/bnc/",
    "tensorboard_logs_path":tensorboard_logs_path,#"./reports/tensorboard/",
    "fast_dev_run":False,
    #"num_sanity_val_steps":True,
    #"overfit_batches":True,
    #"auto_scale_batch_size":"binsearch",
    #"auto_lr_find":True,
    "num_workers":0,
    "min_epochs":100,
    "max_epochs":1000
    }

#TRAIN_DATASET_PATH="./data/processed/b_n_c_256_full/train"
#VALID_DATASET_PATH="./data/processed/b_n_c_256_full/valid"
in_channels=3
classes=3
class_weights=np.array([ 1.05544739, 29.62227178, 53.25918049])

c_light_model = lightningSegModel(in_channels=in_channels, classes=classes,
                                  class_weights=class_weights, **train_params)

data_module = FaultsDataModule()
data_module.setup(dataset_path="./data/processed/f+c_256_Site_A_B_C", encoder_name="resnet101",
                  encoder_weights="imagenet",in_channels=in_channels,
                  classes=classes, batch_size=train_params["batch_size"])



c_light_model.fit(data_module)

dataloader_iterator = iter(data_module.val_dataloader())
sample_val = next(dataloader_iterator)
preds = c_light_model.model(sample_val[0].cuda())
m_preds = torch.argmax(preds,1)
plt.imshow(m_preds[0].cpu())

