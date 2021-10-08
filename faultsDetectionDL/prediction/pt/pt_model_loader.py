#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 20:46:30 2021

@author: cherif
"""

from faultsDetectionDL.prediction.model_loader import GenericModel
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

class PtModel(GenericModel):
    """
    Pytroch model wrapper implementation
    """
    
    def __init__(self, model_path, pretrained_weights, backbone=None, batch_size=16, n_classes=1, device="cuda"):
        GenericModel.__init__(self)
        self.loaded_model=torch.load(model_path, map_location=device)        
        self.pretrained_weights=pretrained_weights
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.preprocesser = smp.encoders.get_preprocessing_fn(backbone, pretrained=pretrained_weights) if backbone else None
        self.device=device
    
    def predict(self, data):
        """
        Run batched prediction on input data.
        Args: 
            data could be ( numpy_array )
                if data is np array, shape should be (img_cnt, X_size, Y_size, band_cnt)
        Returns numpy array with shape (image_cnt, X_size, Y_size, band_cnt)
        """
        image_cnt, X_size, Y_size, band_cnt = data.shape
        
        if self.preprocesser:
            data[:,:,:,0:3] = self.preprocesser(data[:,:,:,0:3])
        
        if type(data) == np.ndarray:
            tensor_x = torch.Tensor(data).permute(0, 3, 1, 2)
            temp_dst = TensorDataset(tensor_x, tensor_x)
            dataloder = DataLoader(temp_dst, self.batch_size, shuffle=False, num_workers=4)
        else:
            raise Exception("Unknown type for pytorch model prediction")
        
        
        preds = np.empty(( image_cnt, 1, X_size, Y_size))
        with torch.no_grad():
            for c_batch_idx , (inputs, labels) in enumerate(tqdm(dataloder, desc="Batch prediction")):
                inputs = inputs.to(self.device)
                output = self.loaded_model(inputs)
                output = output.to(torch.device('cpu'))
                if self.n_classes > 1:
                    output = output.data.numpy().argmax()
                else:
                    output = output.data.numpy()
                preds[self.batch_size*c_batch_idx: self.batch_size*(c_batch_idx+1), :, :, :] = output
            
        #preds_array = np.array(preds)
        
        return np.transpose(preds, (0,2,3,1))
        