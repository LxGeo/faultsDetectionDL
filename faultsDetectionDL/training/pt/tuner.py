#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:01:17 2021

@author: cherif
"""
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import segmentation_models_pytorch as smp
from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity
from faultsDetectionDL.data.pt.faults_dataset import FaultsDataset 
from torch.utils.data import TensorDataset, DataLoader
from segmentation_models_pytorch.utils.losses import BCELoss , JaccardLoss

from faultsDetectionDL.training.pt import TRAIN_PATH,VALID_PATH, IMG_CHANNELS, \
    n_classes, class_weights, activation, model_name_template

config = {
    "LR": tune.loguniform(1e-6, 1e-3),
    "BATCH_SIZE": tune.choice([16, 32]),
    "optim" : tune.choice([
        torch.optim.Adam
        ]),
    "decoder_channels" : tune.choice([ 
        "768,512,256,256,64",
        "512,512,256,256,64",
        "512,512,256,128,64"
        ]),
    "decoder_use_batchnorm": tune.choice([ True, False ]),
    "arch":tune.choice([ smp.Unet, smp.UnetPlusPlus, smp.ResUnet, smp.MAnet ]),
    "encoder_weights" : tune.choice([
        "resnet152==imagenet",
        "resnet101==imagenet",
        "resnext50_32x4d==imagenet",
        "resnext101_32x4d==ssl",
        "resnext101_32x8d==imagenet",
        "resnext101_32x16d==imagenet",
        "se_resnet50==imagenet",
        "se_resnet101==imagenet",
        "se_resnet152==imagenet",
        "se_resnext50_32x4d==imagenet",
        "timm-regnetx_160==imagenet"
        ])
    
}


def train_with_config( config ):
    """
    """
    dc_str = config["decoder_channels"]
    dc_list = list(map(int, dc_str.split(",")))
    arch = config["arch"]
    
    ENCODER, ENCODER_WEIGHTS = config["encoder_weights"].split("==")
    
    #model_name = "Unet_resnext101_32x8d"
    #ENCODER = 'resnext101_32x8d'
    #ENCODER_WEIGHTS = 'imagenet'
    model = arch(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=IMG_CHANNELS,
        classes=n_classes,
        activation=activation,
        decoder_use_batchnorm=config["decoder_use_batchnorm"],
        decoder_channels= dc_list
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_dataset = FaultsDataset(TRAIN_PATH, images_transformations_list, preprocessing=preprocessing_fn, img_bands=IMG_CHANNELS)
    valid_dataset = FaultsDataset(VALID_PATH, images_transformations_list, preprocessing=preprocessing_fn, img_bands=IMG_CHANNELS)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=12)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4)
    
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore()
    ]

    LR = config["LR"]
    optimizer = config["optim"]([ 
        dict(params=model.parameters(), lr=LR),
    ])
    
    total_loss = BCELoss(torch.tensor(class_weights)) + 0.2*JaccardLoss() 
    #loss_name = "weighted_bce_n_jacc"
    
    DEVICE = 'cuda'
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=total_loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=False,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=total_loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=False,
    )
    
    for i in range(1, 3):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        torch.cuda.empty_cache()
        
        tune.report(loss=(valid_logs["l"]), accuracy=valid_logs["iou_score"])



reporter = CLIReporter(
    metric_columns=["loss", "accuracy", "training_iteration"])

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=2,
        grace_period=1,
        reduction_factor=2)
    
result = tune.run(
    train_with_config,
    resources_per_trial={"cpu": 72, "gpu": 2},
    config=config,
    num_samples=50,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
















