#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:42:10 2021

@author: cherif
"""

from faultsDetectionDL.training.ptl.lightning_segmentation_model import lightningSegModel, FaultsDataModule
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl

train_params = {
    "learning_rate":0.00398107,
    "batch_size":16,
    "DEVICE":"gpu",
    "model_outputs_root_path":"./models/bnc/",
    "tensorboard_logs_path":"./reports/tensorboard/",
    "fast_dev_run":False,
    #"num_sanity_val_steps":True,
    #"overfit_batches":True,
    #"auto_scale_batch_size":"binsearch",
    #"auto_lr_find":True,
    "num_workers":72,
    "min_epochs":50,
    "max_epochs":100
    }

TRAIN_DATASET_PATH="./data/processed/b_n_c_256/train"
VALID_DATASET_PATH="./data/processed/b_n_c_256/valid"
in_channels=3
classes=3
class_weights=np.array([ 1.15531204, 10.49434559, 25.54702569])

c_light_model = lightningSegModel(in_channels=in_channels, classes=classes,
                                  class_weights=class_weights, **train_params)

data_module = FaultsDataModule()
data_module.setup(dataset_path="./data/processed/b_n_c_256", encoder_name=c_light_model.encoder_name,
                  encoder_weights=c_light_model.encoder_weights,in_channels=in_channels,
                  classes=classes, batch_size=train_params["batch_size"])

c_light_model.fit(data_module)




sample_val = next(iter(data_module.train_dataloader()))
preds = c_light_model.model.predict(sample_val[0])
preds = torch.argmax(preds,1)
plt.imshow(preds[0])
