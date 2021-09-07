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

class PtModel(GenericModel):
    """
    Pytroch model wrapper implementation
    """
    
    def __init__(self, model_path, backbone=None, batch_size=16, n_classes=1, device="cuda"):
        GenericModel.__init__(self)
        self.loaded_model=torch.load(model_path)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.preprocesser = smp.get_preprocessing(backbone) if backbone else None
        self.device=device
    
    def predict(self, data):
        """
        Run batched prediction on input data.
        Args: 
            data could be ( numpy_array || pytorch dataloader || pytorch tensor)
                if data is np array, shape should be (img_cnt, X_size, Y_size, band_cnt)
        Returns numpy array with shape (image_cnt, X_size, Y_size, band_cnt)
        """
        
        if self.preprocesser:
            data = self.preprocesser(data.copy())
        
        if type(data)==DataLoader:
            dataloder = data
        elif type(data) == torch.Tensor:
            temp_dst = TensorDataset(data, data)
            dataloder = DataLoader(temp_dst, self.batch_size, shuffle=False, num_workers=4)
        elif type(data) == np.ndarray:
            tensor_x = torch.Tensor(data).permute(0, 3, 1, 2)
            temp_dst = TensorDataset(tensor_x, tensor_x)
            dataloder = DataLoader(temp_dst, self.batch_size, shuffle=False, num_workers=4)
        else:
            raise Exception("Unknown type for pytorch model prediction")
        
        
        preds = []
        with torch.no_grad():
            for inputs, labels in dataloder:
                inputs = inputs.to(self.device)
                output = self.loaded_model(inputs)
                output = output.to(torch.device('cpu'))
                if self.n_classes > 1:
                    output = output.data.numpy().argmax()
                else:
                    output = output.data.numpy()
                preds.extend(output)
            
        preds_array = np.array(preds)
        
        return np.transpose(preds_array, (0,2,3,1))
        