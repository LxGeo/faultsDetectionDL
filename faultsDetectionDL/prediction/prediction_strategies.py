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
        out_profile.update(tiled=True, blockxsize=self.pred_size, blockysize=self.pred_size, count=n_classes, dtype= rio.float32)#rio.uint8 if n_classes>1 else rio.float32)
        
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
                
                if not c_patch_array.any():
                    continue
                
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


#####

class BasicStrategy():
    
    def __init__(self, ready_model, rio_dataset, patch_size, n_classes):
        self.model = ready_model
        self.rio_dataset = rio_dataset
        self.patch_size = patch_size
        
        out_profile = rio_dataset.profile.copy()
        out_profile.update(tiled=True, blockxsize=self.patch_size, blockysize=self.patch_size, count=n_classes, dtype= rio.float32)#rio.uint8 if n_classes>1 else rio.float32)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_tempfile:
            self.out_rio_dst = rio.open(out_tempfile.name, mode='w+', **out_profile)
            self.whole_dataset_window = Window(0,0, self.out_rio_dst.width, self.out_rio_dst.height)
    
    def __del__(self):
        self.out_rio_dst.close()
    
        
    
    def fill_out_dataset(self):
        """
        Windowed prediction
        """        
        step_size = 128
        patch_size=self.patch_size
        
        from patchify import patchify, unpatchify
        full_image = reshape_as_image(self.rio_dataset.read())[:,:,:3]
        
        rem_width = (full_image.shape[0]-self.patch_size)%step_size
        rem_height = (full_image.shape[1]-self.patch_size)%step_size
        full_image = np.pad( 
            full_image,
            ((0, patch_size-rem_width), (0, patch_size-rem_height), (0,0)) 
            )
        
        patches = patchify(full_image, (self.patch_size,self.patch_size,3),step_size)
        n_patches = patches.shape[0] * patches.shape[1]
        reshaped_patches = patches.reshape(n_patches, self.patch_size,self.patch_size,3)
                
        patches_preds = self.model.predict(reshaped_patches)
        
        patches_preds = patches_preds.reshape(*patches.shape)     
        
        reconstructed_image = unpatchify(patches_preds, full_image.shape)
        
        reconstructed_image= reconstructed_image[:-(patch_size-rem_width), :-(patch_size-rem_height),:]
        
        self.out_rio_dst.write(reshape_as_raster(reconstructed_image.astype(self.out_rio_dst.meta["dtype"])))
        
        self.out_rio_dst.close()
        self.out_rio_dst = rio.open(self.out_rio_dst.name, "r")


class Basic2Strategy():
    
    def __init__(self, ready_model, rio_dataset, patch_size, n_classes):
        self.model = ready_model
        self.rio_dataset = rio_dataset
        self.patch_size = patch_size
        
        out_profile = rio_dataset.profile.copy()
        out_profile.update(tiled=True, blockxsize=self.patch_size, blockysize=self.patch_size, count=n_classes, dtype= rio.float32)#rio.uint8 if n_classes>1 else rio.float32)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_tempfile:
            self.out_rio_dst = rio.open(out_tempfile.name, mode='w+', **out_profile)
            self.whole_dataset_window = Window(0,0, self.out_rio_dst.width, self.out_rio_dst.height)
    
    def __del__(self):
        self.out_rio_dst.close()
        
    
    def recon_im(self, patches: np.ndarray, im_h: int, im_w: int, n_channels: int, stride: int):
        """
        """
    
        patch_size = patches.shape[1]  # patches assumed to be square
    
        # Assign output image shape based on patch sizes
        rows = ((im_h - patch_size) // stride) * stride + patch_size
        cols = ((im_w - patch_size) // stride) * stride + patch_size
    
        if n_channels == 1:
            reconim = np.zeros((rows, cols))
            divim = np.zeros((rows, cols))
        else:
            reconim = np.zeros((rows, cols, n_channels))
            divim = np.zeros((rows, cols, n_channels))
    
        p_c = (cols - patch_size + stride) / stride  # number of patches needed to fill out a row
    
        totpatches = patches.shape[0]
        initr, initc = 0, 0
    
        # extract each patch and place in the zero matrix and sum it with existing pixel values
    
        reconim[initr:patch_size, initc:patch_size] = patches[0]# fill out top left corner using first patch
        divim[initr:patch_size, initc:patch_size] = np.ones(patches[0].shape)
    
        patch_num = 1
    
        while patch_num <= totpatches - 1:
            if p_c > 1 : initc = initc + stride 
            else : initr = initr + stride
            reconim[initr:initr + patch_size, initc:patch_size + initc] += patches[patch_num]
            divim[initr:initr + patch_size, initc:patch_size + initc] += np.ones(patches[patch_num].shape)
    
            if np.remainder(patch_num + 1, p_c) == 0 and patch_num < totpatches - 1:
                initr = initr + stride
                initc = 0
                reconim[initr:initr + patch_size, initc:patch_size] += patches[patch_num + 1]
                divim[initr:initr + patch_size, initc:patch_size] += np.ones(patches[patch_num].shape)
                patch_num += 1
            patch_num += 1
        # Average out pixel values
        reconstructedim = reconim / divim
    
        return reconstructedim
    
    

    def get_patches(self, GT, stride, patch_size):
        """
        """
        assert (GT.shape[0]-patch_size)%stride==0 and (GT.shape[1]-patch_size)%stride==0, "Check padding settings"
        hr_patches = []
    
        for i in range(0, GT.shape[0]- patch_size + 1, stride):
            for j in range(0, GT.shape[1]- patch_size + 1, stride):
                hr_patches.append(GT[i:i + patch_size, j:j + patch_size])
    
        im_h, im_w = GT.shape[0], GT.shape[1]
    
        if len(GT.shape) == 2:
            n_channels = 1
        else:
            n_channels = GT.shape[2]
    
        patches = np.asarray(hr_patches)
    
        return patches, im_h, im_w, n_channels
        
    
    def fill_out_dataset(self):
        """
        Windowed prediction
        """        
        step_size = self.patch_size//2
        patch_size=self.patch_size
        
        full_image = reshape_as_image(self.rio_dataset.read())[:,:,:3]
        
        rem_width = step_size - (full_image.shape[0]-patch_size)%step_size
        rem_height = step_size - (full_image.shape[1]-patch_size)%step_size
        full_image = np.pad( 
            full_image,
            ((0, rem_width), (0, rem_height), (0,0)) 
            )
                
        patchified_res = self.get_patches(full_image, step_size, patch_size)
                
        patches_preds = self.model.predict(patchified_res[0])
        
        reconstructed_image = self.recon_im(patches_preds, patchified_res[1], patchified_res[2], patchified_res[3], step_size)
        
        reconstructed_image= reconstructed_image[:-(rem_width), :-(rem_height),:]
        
        self.out_rio_dst.write(reshape_as_raster(reconstructed_image.astype(self.out_rio_dst.meta["dtype"])))
        
        self.out_rio_dst.close()
        self.out_rio_dst = rio.open(self.out_rio_dst.name, "r")














            