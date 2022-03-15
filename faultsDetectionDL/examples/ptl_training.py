#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:42:10 2021

@author: cherif
"""

from faultsDetectionDL.training.ptl.lightning_segmentation_model import lightningSegModel
from faultsDetectionDL.data.ptl.faults_data_module import FaultsDataModule
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from ray_lightning import RayPlugin

import ray
#ray.init("auto")

import sys

model_outputs_root_path = sys.argv[1]
tensorboard_logs_path = sys.argv[2]
arch = sys.argv[3]
decoder_channels = [256,256,128,64,32]

train_params = {
    "learning_rate":0.0005,
    "batch_size":8,
    "gpu":True,
    "model_outputs_root_path":model_outputs_root_path,#"./models/bnc/",
    "tensorboard_logs_path":tensorboard_logs_path,#"./reports/tensorboard/",
    "fast_dev_run":False,
    #"num_sanity_val_steps":True,
    #"overfit_batches":True,
    #"auto_scale_batch_size":"binsearch",
    #"auto_lr_find":True,
    "num_workers":72,
    "min_epochs":20,
    "max_epochs":150
    }

#TRAIN_DATASET_PATH="./data/processed/b_n_c_256_full/train"
#VALID_DATASET_PATH="./data/processed/b_n_c_256_full/valid"
in_channels=3
classes=3
class_weights=np.array([ 1.05590573, 29.39718564, 52.82928763])#np.array([1.03848459, 40.085691  , 82.56371167])

c_light_model = lightningSegModel(arch=arch, in_channels=in_channels, classes=classes,
                                  class_weights=class_weights,decoder_channels=decoder_channels, **train_params)
c_light_model=c_light_model.cuda()

data_module = FaultsDataModule()
data_module.setup(dataset_path="./data/processed/clean_f+c_ref_256_Site_A_B_C", encoder_name=c_light_model.encoder_name,
                  encoder_weights=c_light_model.encoder_weights,in_channels=in_channels,
                  classes=classes, batch_size=c_light_model.batch_size)


#trainer = pl.Trainer( plugins=[RayPlugin(num_workers=2, use_gpu=True)] ,**c_light_model._get_trainer_params())
#trainer.fit(c_light_model, datamodule=data_module)


c_light_model.fit(data_module)

dataloader_iterator = iter(data_module.val_dataloader())
sample_val = next(dataloader_iterator)
preds = c_light_model.model.predict(sample_val[0].cuda())
if classes!=1:
    preds = torch.argmax(preds,1)
else:
    preds=preds[:,0,:,:]
plt.imshow(preds[0].cpu())

