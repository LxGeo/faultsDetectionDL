# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:28:55 2022

@author: cherif
"""

import torch
from faultsDetectionDL.training.losses.morphology import Dilation2d, Erosion2d

class ThicknessLoss(torch.nn.Module):
    def __init__(self, feature_width=None):
        # feature_width is maximum
        super(ThicknessLoss, self).__init__()
        assert feature_width >=1, "Feature width must be equal or highrt than 1!"
        
        self.feature_width= feature_width#((feature_width+1) // 2)*2
        
        self.under_erosion = Erosion2d(1,1, self.feature_width, soft_max=False)
        self.over_erosion = Erosion2d(1,1, self.feature_width+1, soft_max=False)
 
    def forward(self, inputs, smooth=1e-5):        
        
        under_eroded = self.under_erosion(inputs)
        over_eroded = self.over_erosion(inputs)
        return 1+(over_eroded.sum() - under_eroded.sum())  / (under_eroded.sum() +smooth)