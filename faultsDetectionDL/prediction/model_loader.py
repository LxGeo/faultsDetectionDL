#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 20:11:35 2021

@author: cherif
"""

import sys

class GenericModel():
    """
    Abstract class that defines models required methods
    """
    _available_frameworks = ("tf", "pt", "ptl")
    preprocesser = None
    loaded_model = None
    
    def preprocess_data(self, data):
        """
        """
        return self.preprocesser(data) if self.preprocesser else data
    
    def predict(self):
        raise NotImplementedError()
    


def get_model_wrapper(framework):
    """
    Returns respective model wrapper class
    """
    
    if framework.lower() == "tf":
        try:
            from faultsDetectionDL.prediction.tf.tf_model_loader import TfModel
            return TfModel
        except ImportError as e:
            print("tensorflow model wrapper could not be imported!\n Check python environment")
            raise(e)
    
    elif framework.lower() == "pt":
        try:
            from faultsDetectionDL.prediction.pt.pt_model_loader import PtModel
            return PtModel
        except ImportError as e:
            print("pytorch model wrapper could not be imported!\n Check python environment")
            raise(e)
    
    elif framework.lower() == "ptl":
        try:
            from faultsDetectionDL.prediction.ptl.ptl_model_loader import PtlModel
            return PtlModel
        except ImportError as e:
            print("pytorch model wrapper could not be imported!\n Check python environment")
            raise(e)
            
    else:
        "framework name not found in set: {}".format(GenericModel._available_frameworks)
        sys.exit(1)

