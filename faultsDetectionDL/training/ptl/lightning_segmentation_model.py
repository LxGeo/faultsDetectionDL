#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 13:59:23 2021

@author: cherif
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity
from faultsDetectionDL.data.pt.faults_dataset import FaultsDataset 
import torch
import numpy as np
import os
import shutil


def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    #valid_pixel_mask = true.ne(255)  # valid pixel mask
    #true = true.masked_select(valid_pixel_mask).to("cpu")
    #pred = pred.masked_select(valid_pixel_mask).to("cpu")
    true= true.to("cpu")
    pred= pred.to("cpu")
    
    pred = torch.argmax(pred,1)
    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()


class FaultsDataModule(pl.LightningDataModule):

    def setup(self, dataset_path, encoder_name, encoder_weights, in_channels, classes, batch_size):
                  
        self.train_dataset_path=os.path.join(dataset_path, 'train')
        self.valid_dataset_path=os.path.join(dataset_path, 'valid')
        self.batch_size=batch_size
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
        self.train_dataset = FaultsDataset(self.train_dataset_path, images_transformations_list,
                                           preprocessing=self.preprocessing_fn, img_bands=in_channels,
                                           num_classes=classes)
        self.valid_dataset = FaultsDataset(self.valid_dataset_path, [Trans_Identity()],
                                           preprocessing=self.preprocessing_fn, img_bands=in_channels,
                                           num_classes=classes)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=72)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size)


class lightningSegModel(pl.LightningModule):
    
    def __init__(self, arch="Unet", encoder_name="resnext101_32x8d",
                 encoder_weights="imagenet", in_channels=3, classes=1, class_weights=None, loss=torch.nn.CrossEntropyLoss,
                 learning_rate=1e-5, **kwargs):
        super(lightningSegModel, self).__init__()
        self.arch=arch
        self.encoder_name=encoder_name
        self.encoder_weights=encoder_weights                        
        # Create model
        self.model = smp.create_model(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=classes)
        self.learning_rate=learning_rate
        print(self.learning_rate)
        self.class_weights=torch.tensor(class_weights).float()
        self.loss=loss(self.class_weights)
        
        # train params
        model_outputs_root_path= kwargs.get("model_outputs_root_path", "./models")
        tensorboard_logs_root_path= kwargs.get("tensorboard_logs_root_path", "./reports/tensorboard/")
        trial_name = "_".join([arch, encoder_name])
        self.output_path = os.path.join(model_outputs_root_path, trial_name)
        self.log_path = os.path.join(tensorboard_logs_root_path, trial_name)
        for dir_path in [self.output_path, self.log_path]:
            #if os.path.exists(dir_path):
            #    shutil.rmtree(dir_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        self.gpu= kwargs.get("gpu", torch.cuda.is_available())
        self.num_workers=kwargs.get("num_workers",1)
        self.patience= kwargs.get("patience", 4)
        self.min_epochs= kwargs.get("min_epochs", 6)
        self.max_epochs= kwargs.get("max_epochs", 50)
        self.batch_size= kwargs.get("batch_size", 8)
        self.overfit_batches=kwargs.get("overfit_batches",False)
        self.fast_dev_run=kwargs.get("fast_dev_run", False)
        self.auto_scale_batch_size=kwargs.get("auto_scale_batch_size",None)
        self.auto_lr_find=kwargs.get("auto_lr_find",None)
        
        # Track validation IOU globally (reset each epoch)
        self.intersection = 0
        self.union = 0
        
    def forward(self, image):
        # Forward pass
        return self.model(image)
    
    
    def calc_loss(self, logits, labels):
        return self.loss(logits, labels)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.calc_loss(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.calc_loss(logits, y)
        self.log('val_loss', loss)
        
        preds = torch.softmax(logits, dim=1)#[:, 1]
        preds = (preds > 0.5) * 1
        intersection, union = intersection_and_union(preds, y)
        self.intersection += intersection
        self.union += union

        # Log batch IOU
        batch_iou = intersection / union
        self.log(
            "iou", batch_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return batch_iou
    
    def validation_epoch_end(self, outputs):
        # Reset metrics before next epoch
        self.intersection = 0
        self.union = 0
            
    
    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience
        )
        scheduler = {
            "scheduler": scheduler, "interval": "epoch", "monitor": "iou_epoch",
        }  # logged value to monitor
        return [optimizer], [scheduler]
    
    def _get_trainer_params(self):
        # Define callback behavior
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="iou_epoch",
            mode="max",
            verbose=True,
        )
        # Define early stopping behavior
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="iou_epoch",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )
        # Specify where TensorBoard logs will be saved
        logger = pl.loggers.TensorBoardLogger(self.log_path, name="benchmark-model")
        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "default_root_dir": self.output_path,
            "logger": logger,
            "gpus": None if not self.gpu else 1,
            "fast_dev_run": self.fast_dev_run,
            #"num_sanity_val_steps": self.val_sanity_checks,
            "overfit_batches":self.overfit_batches,
            "auto_scale_batch_size":self.auto_scale_batch_size,
            "auto_lr_find":self.auto_lr_find
        }
        return trainer_params
    
    def fit(self, data_module):
        # Set up and fit Trainer object
        self.trainer = pl.Trainer(**self._get_trainer_params())
        self.trainer.fit(self, datamodule=data_module)
    
    def tune(self, data_module):
        self.trainer = pl.Trainer(**self._get_trainer_params())
        self.trainer.tune(self, datamodule=data_module)

