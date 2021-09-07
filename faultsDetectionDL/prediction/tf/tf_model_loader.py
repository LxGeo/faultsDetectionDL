#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:32:43 2021

@author: cherif
"""

from keras.models import load_model
from faultsDetectionDL.prediction.model_loader import GenericModel
import segmentation_models as sm
import numpy as np


class TfModel(GenericModel):
    """
    Tensorflow model wrapper implementation
    """
    
    def __init__(self, model_path, backbone=None, batch_size=16, n_classes=1, custom_objects=None):
        GenericModel.__init__(self)
        
        self.custom_objects = custom_objects
        self.loaded_model=load_model(model_path,self.custom_objects)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.preprocesser = sm.get_preprocessing(backbone) if backbone else None
    
    def predict(self, data):
        """
        Args:
            data: numpy array
        """        
        if self.preprocesser:
            data = self.preprocesser(data.copy())
        
        preds_array = self.loaded_model.predict(data,batch_size=self.batch_size)
        if self.n_classes > 1 :
            preds_array = np.argmax(preds_array, axis=-1)
            preds_array = np.expand_dims(preds_array, axis=-1)                
        return preds_array
        