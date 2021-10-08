#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:44:10 2021

@author: cherif
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ReflectionPad2d
from segmentation_models_pytorch.utils import base
from numba import jit


class neighborLoss(base.Loss):

    def __init__(self, neighbor_len = 2, central_value=128, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_len = neighbor_len
        
        self.EPS = 1e-3
        ##kernel creation
        
        ones_kernel = np.ones( (1+2*neighbor_len, 1+2*neighbor_len) )
        ones_kernel[neighbor_len, neighbor_len] = central_value
        
        self.axis_pads = ( neighbor_len,neighbor_len, neighbor_len,neighbor_len )
        
        self.kernel = ones_kernel/ones_kernel.sum()
        self.kernel_flat = torch.tensor(self.kernel.flatten(), device="cuda")
        self.ones_mat = torch.ones(self.kernel.shape, device="cuda")
        
        self.reflection_pad = ReflectionPad2d(neighbor_len)
    
    
    def forward(self, pred, gt):
        
        sample_cnt =pred.shape[0]
        
        pred_clip = torch.clip(pred, self.EPS, 1-self.EPS)
        gt_clip = gt
        
        
        all_losses = torch.empty(pred_clip.size())
        
        for sample_i in range(sample_cnt):
        
            padded_gt = self.reflection_pad(gt_clip[sample_i:sample_i+1, :,:,:])
            padded_pred = self.reflection_pad(pred_clip[sample_i:sample_i+1, :,:,:])
        
            neighborLoss_func(padded_pred, padded_gt, self.neighbor_len, self.kernel_flat, self.reflection_pad, self.ones_mat, all_losses, sample_i)
        
        return torch.sum(all_losses)/gt_clip.size 
    
    """
    def forward(self, pred, gt):
        
        
        sample_cnt =pred.shape[0]
        
        pred_clip = torch.clip(pred, self.EPS, 1-self.EPS)
        gt_clip = gt
        
        
        all_losses = torch.empty(pred_clip.size())
        
        for sample_i in range(sample_cnt):
        
            padded_gt = self.reflection_pad(gt_clip[sample_i:sample_i+1, :,:,:])
            padded_pred = self.reflection_pad(pred_clip[sample_i:sample_i+1, :,:,:])
            
            for _ix in range(self.neighbor_len, padded_gt.shape[-2]-self.neighbor_len):
                for _iy in range(self.neighbor_len, padded_gt.shape[-1]-self.neighbor_len):
                    
                    gt_view= padded_gt[0,0,_ix, _iy] * self.ones_mat
                    pred_view = padded_pred[0,0,_ix-self.neighbor_len : _ix+self.neighbor_len+1, _iy-self.neighbor_len: _iy+self.neighbor_len+1]
                    
                    gt_view_flat = gt_view.flatten()
                    pred_view_flat = pred_view.flatten()
                    
                    log_proba_flattened = ( gt_view_flat * torch.log(pred_view_flat) ) \
                        + ( (1-gt_view_flat) * torch.log(1-pred_view_flat) )
                                        
                    all_losses[sample_i, 0, _ix-self.neighbor_len, _iy-self.neighbor_len]= -torch.sum( log_proba_flattened * self.kernel_flat )
                    
            
        return torch.sum(all_losses)/gt_clip.size
"""

@jit(nopython=True, parallel=True, fastmath=True)
def neighborLoss_func(padded_pred, padded_gt, neighbor_len, kernel_flat, reflection_pad, ones_mat, all_losses, sample_i):
    
    
    for _ix in range(neighbor_len, padded_gt.shape[-2]-neighbor_len):
        for _iy in range(neighbor_len, padded_gt.shape[-1]-neighbor_len):
            
            gt_view= padded_gt[0,0,_ix, _iy] * ones_mat
            pred_view = padded_pred[0,0,_ix-neighbor_len : _ix+neighbor_len+1, _iy-neighbor_len: _iy+neighbor_len+1]
            
            gt_view_flat = gt_view.flatten()
            pred_view_flat = pred_view.flatten()
            
            log_proba_flattened = ( gt_view_flat * torch.log(pred_view_flat) ) \
                + ( (1-gt_view_flat) * torch.log(1-pred_view_flat) )
                
            all_losses[sample_i, 0, _ix-neighbor_len, _iy-neighbor_len]= -torch.sum( log_proba_flattened * kernel_flat )
            
    