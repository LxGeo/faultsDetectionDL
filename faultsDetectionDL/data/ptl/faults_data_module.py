#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 17:45:24 2022

@author: cherif
"""
import os
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from faultsDetectionDL.utils.image_transformation import images_transformations_list, Trans_Identity
from faultsDetectionDL.data.pt.faults_dataset import FaultsDataset 

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