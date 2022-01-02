#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 21:01:19 2021

@author: cherif
"""


"""
class PredictionStrategy:
    def __init__(self, ready_model, rio_dataset, patch_size):
        self.model = ready_model
        self.rio_dataset = rio_dataset
        self.patch_size = patch_size

    def map(self):
        raise NotImplementedError

    def reduce(self, other):
        raise NotImplementedError
"""

import rasterio as rio
import tempfile
import numpy as np
import geopandas as gpd
import math
from rasterio.windows import Window
from shapely.geometry import box
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio import mask
from tqdm import tqdm


class PatchOverlapStrategy():
    
    def __init__(self, ready_model, rio_dataset, patch_size, n_classes, overlap_ratio=0.5):
        self.model = ready_model
        self.rio_dataset = rio_dataset
        self.patch_size = patch_size
        self.overlap_ratio=overlap_ratio
        self.pred_size = round(patch_size * overlap_ratio)
        self.pad_count = (self.patch_size-self.pred_size)//2
        
        out_profile = rio_dataset.profile.copy()
        out_profile.update(tiled=True, blockxsize=self.pred_size, blockysize=self.pred_size, count=1, dtype= rio.uint8 if n_classes>1 else rio.float32)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_tempfile:
            self.out_rio_dst = rio.open(out_tempfile.name, mode='w+', **out_profile)
            self.whole_dataset_window = Window(0,0, self.out_rio_dst.width, self.out_rio_dst.height)
    
    def __del__(self):
        self.out_rio_dst.close()
    
    
    def create_patches_and_windows(self):
        """
        
        """
        padded_array = np.pad( 
            reshape_as_image(self.rio_dataset.read()),
            ((self.pad_count, self.patch_size), (self.pad_count, self.patch_size), (0,0)) 
            )
                
        IMG_HEIGHT = self.rio_dataset.height
        IMG_WIDTH = self.rio_dataset.width
    
        # Define the grid blocks
        Nx_blocks = math.ceil(IMG_WIDTH/self.pred_size)
        Ny_blocks = math.ceil(IMG_HEIGHT/self.pred_size)
    
        big_patches_array=[] ## 100% patch size
        small_windows=[] ## overlap_ratio % patch size
        
        for ix in range(Nx_blocks):
            for iy in range(Ny_blocks):                
                
                c_patch_array = padded_array[iy*self.pred_size : iy*self.pred_size+self.patch_size,
                                              ix*self.pred_size : ix*self.pred_size+self.patch_size, :]
                
                pred_win = Window(ix*self.pred_size, iy*self.pred_size, self.pred_size, self.pred_size)     
                
                big_patches_array.append(c_patch_array)
                small_windows.append(pred_win)
        
        big_patches_array=np.array(big_patches_array)
        return big_patches_array,small_windows
    
    def fill_out_dataset(self):
        """
        Windowed prediction
        """        
        
        big_patches_array,small_windows = self.create_patches_and_windows()
        
        ### predict on all patches
        # !! if any error below try rio interoperability
        
        big_patches_preds = self.model.predict(big_patches_array)
        small_patches_preds = big_patches_preds[:, self.pad_count:-self.pad_count,self.pad_count:-self.pad_count,: ]
        
        
        for patch_index ,c_patch_window in enumerate(tqdm(small_windows, desc="Writing prediction")):
            # out of range case check
            intersection_window = c_patch_window.intersection(self.whole_dataset_window)
            array_to_save = small_patches_preds[patch_index, :intersection_window.width, :intersection_window.height, :]
            
            self.out_rio_dst.write(reshape_as_raster(array_to_save.astype(self.out_rio_dst.meta["dtype"])), window = intersection_window)
        
        self.out_rio_dst.close()
        self.out_rio_dst = rio.open(self.out_rio_dst.name, "r")



















            