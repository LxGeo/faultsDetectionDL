#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 17:48:56 2022

@author: cherif
"""

from faultsDetectionDL.prediction.model_loader import GenericModel
from faultsDetectionDL.training.ptl.lightning_segmentation_model import lightningSegModel
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

class PtlModel(GenericModel):
    """
    Pytroch model wrapper implementation
    """
    
    def __init__(self, model_path, device="cuda", **kwargs):
        GenericModel.__init__(self)
        self.loaded_model=lightningSegModel.load_from_checkpoint(model_path)        
        self.batch_size = self.loaded_model.batch_size
        self.n_classes = self.loaded_model.n_classes
        self.in_channels = self.loaded_model.in_channels
        self.preprocesser = smp.encoders.get_preprocessing_fn(self.loaded_model.encoder_name, self.loaded_model.encoder_weights)
        self.device=device
        self.loaded_model.to(device)
    
    def predict(self, data):
        """
        Run batched prediction on input data.
        Args: 
            data could be ( numpy_array )
                if data is np array, shape should be (img_cnt, X_size, Y_size, band_cnt)
        Returns numpy array with shape (image_cnt, X_size, Y_size, band_cnt)
        """
        image_cnt, X_size, Y_size, band_cnt = data.shape
        if band_cnt != self.in_channels:
            data = data[:,:,:,:self.in_channels]
            band_cnt = self.in_channels
        
        if self.preprocesser:
            #data[:,:,:,0:3] = self.preprocesser(data[:,:,:,0:3])
            data = self.preprocesser(data)
        
        if type(data) == np.ndarray:
            tensor_x = torch.Tensor(data).permute(0, 3, 1, 2)
            temp_dst = TensorDataset(tensor_x, tensor_x)
            dataloder = DataLoader(temp_dst, self.batch_size, shuffle=False, num_workers=1)
        else:
            raise Exception("Unknown type for pytorch model prediction")
        
        
        preds = np.empty(( image_cnt, 1, X_size, Y_size))
        with torch.no_grad():
            for c_batch_idx , (inputs, labels) in enumerate(tqdm(dataloder, desc="Batch prediction")):
                inputs = inputs.to(self.device)
                output = self.loaded_model(inputs)
                output = output.to(torch.device('cpu'))
                if self.n_classes > 1:
                    output = np.expand_dims(output.numpy().argmax(axis=1), axis=1)
                else:
                    output = output.data.numpy()
                
                preds[self.batch_size*c_batch_idx: self.batch_size*(c_batch_idx+1), :, :, :] = output
            
        #preds_array = np.array(preds)
        
        return np.transpose(preds, (0,2,3,1))