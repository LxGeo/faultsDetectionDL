#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 13:59:23 2021

@author: cherif
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from faultsDetectionDL.training.registereis import optimizers_registery, loss_registery, schedulers_registry
from functools import partial
import faultsDetectionDL.training.metrics as smp_metrics

from matplotlib import pyplot as plt
import torch
import torchvision
import numpy as np
import os


def intersection_and_union(pred, true, n_classes, threshold=0.5):
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
    if n_classes !=1:
        pred = torch.argmax(pred,1)    
        ##on hot encode    
        h_true=torch.nn.functional.one_hot(true,n_classes)
        h_pred=torch.nn.functional.one_hot(pred,n_classes)    
        # Intersection and union totals
        intersection = torch.logical_and(h_true, h_pred)
        union = torch.logical_or(h_true, h_pred)
        return intersection.sum(), union.sum()
    else:
        pred = (pred > threshold).long()
        intersection = torch.logical_and(true[:,1:,:,:], pred[:,1:,:,:])
        union = torch.logical_or(true[:,1:,:,:], pred[:,1:,:,:])
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
    
    def __init__(self, arch, encoder_name,encoder_weights, in_channels=3, classes=1, lr=1.0e-4,
                 decoder_channels=[512,512,256,128,64] , class_weights=None, decoder_use_batchnorm=False, decoder_attention_type=None, **kwargs):
                
        super(lightningSegModel, self).__init__()
        self.save_hyperparameters()
        
        self.cfg = kwargs
        self.arch=self.hparams.arch
        self.encoder_name=self.hparams.encoder_name
        self.encoder_weights=self.hparams.encoder_weights
        self.decoder_channels=self.hparams.decoder_channels                     
        # Create model
        self.decoder_use_batchnorm = self.hparams.get("decoder_use_batchnorm")
        self.decoder_attention_type = self.hparams.get("decoder_attention_type")
        self.seghead_dropout = self.hparams.get("dropout")
        self.model = smp.create_model(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=classes, decoder_channels=self.decoder_channels,
                                      encoder_depth=len(self.decoder_channels),
                                      decoder_use_batchnorm=decoder_use_batchnorm)#, dropout=self.seghead_dropout)#, decoder_attention_type=decoder_attention_type)
        self.n_classes=self.hparams.classes
        self.in_channels=self.hparams.in_channels
        
        self.lr=self.hparams.lr
        
        self.class_weights=torch.tensor(self.hparams.class_weights).float().cuda() if (self.hparams.class_weights is not None) else None
        
        if self.hparams.get("class_weights_metric") is not None:
            reduction_mode, class_weights = "weighted", self.hparams.class_weights_metric
        else:
            reduction_mode, class_weights = "micro", None
        metric_args = {
            "reduction":reduction_mode, "class_weights":class_weights, "zero_division":0
        }
        
        self.metrics_callable = {
            "iou_score" : partial(smp_metrics.iou_score, **metric_args),
            "f1_score" : partial(smp_metrics.f1_score, **metric_args),
            "f2_score" : partial(smp_metrics.fbeta_score, beta=2, **metric_args),
            "accuracy" : partial(smp_metrics.accuracy, **metric_args),
            "recall" : partial(smp_metrics.recall, **metric_args),
            "precision" : partial(smp_metrics.precision, **metric_args),
        }
        
        self.losses_names = []
        self.losses_weight = []
        self.losses_callable = [ ]
        for c_loss_obj in self.cfg["LOSSES"]:
            c_loss_name = c_loss_obj.NAME; self.losses_names.append(c_loss_name)  
            c_loss_weight = c_loss_obj.WEIGHT; self.losses_weight.append(c_loss_weight)  
            c_loss_kwargs = vars(c_loss_obj.ARGS) if c_loss_obj.ARGS else {}
                
            c_raw_loss = loss_registery.get(c_loss_name)
            if "ce_loss" in c_loss_name: #temporary
                c_loss_kwargs["weight"]=self.class_weights       
            self.losses_callable.append(
                c_raw_loss( **c_loss_kwargs )
            )
        
        self.softmax = torch.nn.Softmax(dim=1)
        
        
    def forward(self, image):
        # Forward pass
        image_out = self.model(image)
        return image_out
    
    def combined_loss(self, logits, labels):
        """
        """
        seperate_losses = []
        for c_l in self.losses_callable:
            seperate_losses.append(c_l(logits, labels))
        total_loss = sum([ c_w * c_l for c_w, c_l in zip(self.losses_weight, seperate_losses) ]) / sum(self.losses_weight)
        return total_loss, seperate_losses
    
    def calc_loss(self, logits, labels):
        return self.combined_loss(logits, labels)
    
    def calc_metrics_scores(self, logits, labels):        
        
        tp, fp, fn, tn = smp_metrics.get_stats(torch.softmax(logits,1), labels.long(), mode='multilabel', threshold=0.5)
        scores = {
            metric_key: metric_callable(tp, fp, fn, tn).item() for metric_key, metric_callable in self.metrics_callable.items()
        }
        return scores
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        tot_loss, seperate_losses = self.calc_loss(logits, y)
        
        for loss_name, loss_value in zip(self.losses_names, seperate_losses):
            self.log(f'train_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        self.log('train_loss', tot_loss, on_step=True, on_epoch=True, prog_bar=True)
        return tot_loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        tot_loss, seperate_losses = self.calc_loss(logits, y)
        
        for loss_name, loss_value in zip(self.losses_names, seperate_losses):
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        self.log('val_loss', tot_loss, on_step=True, on_epoch=True)
        return self.calc_metrics_scores(logits, y[0])

    def predict_step(self, batch, batch_idx=None, dataloader_idx= None):
        return self.forward(batch[0])
    
    def image_grid(self, x, y, preds, prefix):  
        
        preds = self.softmax(preds)
        classes = torch.argmax(preds,1)
        bs = preds.shape[0]
                
        for i in range(bs):
            fixed_rgb = (x[i]-x[i].min())/(x[i].max()-x[i].min())
            self.logger.experiment.add_image(prefix+"rgb_{}".format(i),fixed_rgb,self.current_epoch)
            fixed_gt = y[0][i]
            self.logger.experiment.add_image(prefix+"gt_{}".format(i),fixed_gt,self.current_epoch)
            fixed_preds = (preds[i]-preds[i].min())/(preds[i].max()-preds[i].min())
            self.logger.experiment.add_image(prefix+"preds_{}".format(i),fixed_preds,self.current_epoch)
            fixed_preds_class = torch.unsqueeze(classes[i],0)/self.n_classes
            self.logger.experiment.add_image(prefix+"preds_class_{}".format(i),fixed_preds_class,self.current_epoch)
    
    def validation_epoch_end(self, outputs):
        
        if hasattr(self, "sample_val"):
            X,Y = self.sample_val
            logits = self.forward(X.to(self.device))            
            self.image_grid(X, Y, logits, "val_")
        if hasattr(self, "sample_train"):
            X,Y = self.sample_train
            logits = self.forward(X.to(self.device))            
            self.image_grid(X, Y, logits, "train_")
        
        combined_scores = {
            metric_key: np.mean([x[metric_key] for x in outputs])
            for metric_key in outputs[0]
        }
        for metric_key, metric_value in combined_scores.items():
            self.log(f"val_{metric_key}", metric_value, on_epoch=True)
        
        return {"log_scores":combined_scores}
    
    def get_preprocessing_fn(self):
        return smp.encoders.get_preprocessing_fn(self.encoder_name, self.encoder_weights)

    def configure_optimizers(self):
        
        optimizer_params = vars(self.cfg["OPTIMIZER"].PARAMS)
        optimizer_params["lr"] = self.lr
        optimizer = optimizers_registery.get(self.cfg["OPTIMIZER"].NAME)(
            self.model.parameters(), **optimizer_params
        )

        schedulers = []
        if "SCHEDULER" in self.cfg and self.cfg["SCHEDULER"].USE:
            sched = schedulers_registry.get(self.cfg["SCHEDULER"].NAME)
            if self.cfg["SCHEDULER"].PARAMS is not None:
                scheduler_params = vars(self.cfg["SCHEDULER"].PARAMS)
                scheduler = sched(optimizer, **scheduler_params)
            else:
                scheduler = sched(optimizer)
            
            schedulers.append(scheduler)
        
        return [optimizer], schedulers


