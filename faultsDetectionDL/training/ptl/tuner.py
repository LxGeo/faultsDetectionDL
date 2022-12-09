#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:08:56 2022

@author: cherif
"""

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from faultsDetectionDL.training.ptl.lightning_segmentation_model import lightningSegModel
from faultsDetectionDL.data.ptl.faults_data_module import FaultsDataModule
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from functools import partial


from ray.tune.integration.pytorch_lightning import TuneReportCallback

data_module = FaultsDataModule()
data_module.setup(dataset_path="/home/cherif/Documents/LxGeo/faultsDetectionDL/data/processed/clean_f+c_ref_256_Site_A_B_C", encoder_name="resnet18",
                  encoder_weights="imagenet",in_channels=3,
                  classes=3, batch_size=39)
dataloader_iterator = iter(data_module.val_dataloader())
sample_val = next(dataloader_iterator)

decoder_channels_map = {
    1:[256,256,128,64,32],
    2:[512,256,128,64,32],
    3:[128,128,128,64,32]
    }

def train_model_hook_with_args(config, sample_val, decoder_channels_map):
    
    config_hash = "lr={}_bs={}_optim={}_dec_ch={}".format(config["LR"], config["BATCH_SIZE"], 
                                                              config["optim"].__name__, str(decoder_channels_map[config["dec_ch"]]))
    model_outputs_root_path = "/home/cherif/Documents/LxGeo/faultsDetectionDL/models/faults/tune4/"+config_hash
    tensorboard_logs_path = "/home/cherif/Documents/LxGeo/faultsDetectionDL/reports/tensorboard/faults/tune4/"
    arch = "Unet"
    decoder_channels = decoder_channels_map[config["dec_ch"]]
    
    encoder_name = "resnet18"
    encoder_weights = "imagenet"

    train_params = {
        "learning_rate":config["LR"],
        "batch_size":config["BATCH_SIZE"],
        "gpu":True,
        "model_outputs_root_path":model_outputs_root_path,#"./models/bnc/",
        "tensorboard_logs_path":tensorboard_logs_path,#"./reports/tensorboard/",
        "fast_dev_run":False,
        "decoder_use_batchnorm":True,
        "num_workers":72
        }
    
    in_channels=3
    classes=3
    class_weights=1/np.array([ 1.05590573, 29.39718564, 52.82928763])

    c_light_model = lightningSegModel(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes,
                                      class_weights=class_weights,decoder_channels=decoder_channels,
                                      optim = config["optim"],**train_params)
    
    c_light_model=c_light_model.cuda()
    
    # data loading
    data_module = FaultsDataModule()
    data_module.setup(dataset_path="/home/cherif/Documents/LxGeo/faultsDetectionDL/data/processed/clean_f+c_ref_256_Site_A_B_C", encoder_name=c_light_model.encoder_name,
                      encoder_weights=c_light_model.encoder_weights,in_channels=in_channels,
                      classes=classes, batch_size=c_light_model.batch_size)
        
    c_light_model.sample_val = sample_val
    
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    
    trainer = pl.Trainer(min_epochs = 20, max_epochs=40, callbacks=callbacks)    
    
    trainer = pl.Trainer(**c_light_model._get_trainer_params())
    trainer.fit(c_light_model, datamodule=data_module)
    torch.cuda.empty_cache()


config = {
    "LR": tune.uniform(0.00001, 0.0005),
    "BATCH_SIZE": tune.choice([16, 36, 40, 42]),
    "optim" : tune.choice([
        torch.optim.AdamW
        ]),
    "dec_ch":tune.choice([1,2,3])
    
}

train_model_hook = partial(train_model_hook_with_args, sample_val=sample_val, decoder_channels_map=decoder_channels_map)

analysis = tune.run(
        train_model_hook,
        metric="loss",
        mode="min",
        resources_per_trial={"cpu": 72, "gpu": 2},
        config=config,
        num_samples=50,
        name="tune lsegmodel")
