#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 13:59:23 2021

@author: cherif
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import torch
import numpy as np
import os
import shutil


def intersection_and_union(pred, true, n_classes):
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
    #true= true.to("cpu")
    #pred= pred.to("cpu")    
    pred = torch.argmax(pred,1)    
    ##on hot encode    
    h_true=torch.nn.functional.one_hot(true,n_classes)
    h_pred=torch.nn.functional.one_hot(pred,n_classes)    
    # Intersection and union totals
    intersection = torch.logical_and(h_true, h_pred)
    union = torch.logical_or(h_true, h_pred)
    return intersection.sum(), union.sum()

def weighted_iou_loss(pred, true, class_weights):
    
    pred = torch.argmax(pred,1)    
    ##on hot encode    
    h_true=torch.nn.functional.one_hot(true,len(class_weights))
    h_pred=torch.nn.functional.one_hot(pred,len(class_weights))  
    
    # Intersection and union totals
    intersection = torch.logical_and(h_true, h_pred)    
    union = torch.logical_or(h_true, h_pred)
    # height and width sum()
    intersection = intersection.sum(-2).sum(-2)
    union = union.sum(-2).sum(-2)
    
    # batch_class wise loss tensor
    b_cl_loss = ( intersection / (union + 1e-5) )
    weighted_loss = b_cl_loss#*class_weights
    
    return weighted_loss.mean()


class lightningSegModel(pl.LightningModule):
    
    def __init__(self, arch="Unet", encoder_name="resnext101_32x8d",
                 encoder_weights="imagenet", in_channels=3, classes=1, decoder_channels=[512,512,256,128,64] , class_weights=None,
                 learning_rate=1e-5, **kwargs):
        super(lightningSegModel, self).__init__()
        self.save_hyperparameters()
        self.arch=arch
        self.encoder_name=encoder_name
        self.encoder_weights=encoder_weights                        
        # Create model
        self.model = smp.create_model(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=classes, decoder_channels=decoder_channels)
        self.n_classes=classes
        self.in_channels=in_channels
        self.learning_rate=learning_rate
        
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
        self.num_workers=kwargs.get("num_workers",72)
        self.patience= kwargs.get("patience", 4)
        self.min_epochs= kwargs.get("min_epochs", 6)
        self.max_epochs= kwargs.get("max_epochs", 50)
        self.batch_size= kwargs.get("batch_size", 8)
        self.overfit_batches=kwargs.get("overfit_batches",False)
        self.fast_dev_run=kwargs.get("fast_dev_run", False)
        self.auto_scale_batch_size=kwargs.get("auto_scale_batch_size",None)
        self.auto_lr_find=kwargs.get("auto_lr_find",None)
        
        
        self.class_weights=torch.tensor(class_weights).float().cuda() if (class_weights is not None) else None
        # Track validation IOU globally (reset each epoch)
        self.intersection = 0
        self.union = 0
        self.jacc = smp.losses.JaccardLoss("multiclass")
        self.cce = torch.nn.CrossEntropyLoss(self.class_weights)
        self.cce_loss_weight = 0.9
        
    def forward(self, image):
        # Forward pass
        return self.model(image)
    
    def combined_loss(self, logits, labels):
        """
        CCE and IOU
        """        
        #loss_iou = weighted_iou_loss(logits, labels, self.class_weights)
        loss_cce = self.cce(logits, labels)
        loss_iou = self.jacc(logits, labels)
        total_loss = self.cce_loss_weight * loss_cce + (1-self.cce_loss_weight)*loss_iou
        return total_loss
            
    def calc_loss(self, logits, labels):
        return self.combined_loss(logits, labels)
    
    def training_step(self, train_batch, batch_idx):
        self.log("cce_loss_weight", self.cce_loss_weight, on_step=False, on_epoch=True)
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
        # ref image
        if (batch_idx%100==0):
            self.reference_image = torch.argmax(logits,1)[0]
            self.showActivations(self.reference_image)
            
        
        #preds = torch.softmax(logits, dim=1)#[:, 1]
        #preds = (preds > 0.5) * 1
        intersection, union = intersection_and_union(logits, y, self.n_classes)
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
        # update cce weight every n epoch        
        if self.current_epoch % 3 ==0:
            self.cce_loss_weight = self.cce_loss_weight/2
    
    
    def showActivations(self,x):
            # logging reference image    
            self.logger.experiment.add_image("output_",torch.Tensor.cpu(x),self.current_epoch,dataformats="HW")
            
    
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
            save_top_k=4,
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

