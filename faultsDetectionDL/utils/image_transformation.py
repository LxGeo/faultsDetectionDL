#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:47:33 2021

@author: Bilel Kanoun
"""
from skimage.transform import rotate
import numpy as np

class Image_Transformation:
    
    def apply_transformation(image,gt):
        pass
    
class Trans_Rot90(Image_Transformation):
    def apply_transformation(image1, gt1):
        image1_t = rotate(image1, 90)
        gt1_t = rotate(gt1, 90)
        
        return (image1_t, gt1_t)

class Trans_Rot180(Image_Transformation):
    def apply_transformation(image1, gt1):
        image1_t = rotate(image1, 180)
        gt1_t = rotate(gt1, 180)
        
        return (image1_t, gt1_t)

class Trans_Rot270(Image_Transformation):
    def apply_transformation(image1, gt1):
        image1_t = rotate(image1, 270)
        gt1_t = rotate(gt1, 270)
        
        return (image1_t, gt1_t)

class Trans_Flipud(Image_Transformation):
    def apply_transformation(image1, gt1):
        image1_t = np.flipud(image1)
        gt1_t = np.flipud(gt1)
        
        return (image1_t, gt1_t)
    
class Trans_fliplr(Image_Transformation):
    def apply_transformation(image1, gt1):
        image1_t = np.fliplr(image1)
        gt1_t = np.fliplr(gt1)
        
        return (image1_t, gt1_t)

class Trans_gaussian_noise(Image_Transformation):
    def apply_transformation(image1, gt1):
        noise = np.random.normal(loc = 0.0, scale = 1, size = image1.shape) 
        image1_t = np.clip(image1+noise, 0 ,255)
        
        return (image1_t, gt1)


images_transformations_list=[Trans_Rot180, Trans_Flipud, Trans_fliplr, Trans_gaussian_noise]
TDO = { images_transformations_list[i].__name__ : i for i in range(len(images_transformations_list)) }
to_ignore_trans_lists=[ [0,1,2] ]

class recurse_transform():
    
    _images_transformations_list = images_transformations_list
    
    def __init__(self, image_couple, max_trans=20):
        self.image_couple = image_couple
        self.max_trans = max_trans
        self.all_transformed = []
        self.trans_indices=[]
        
    def run_recurse(self, couple_image_gt, c_Ts, names_idx):
        """
        """
        if len(c_Ts)==1:
            c_names_indices = names_idx+[TDO[c_Ts[0].__name__]]
            if any([all([ c in c_ignore_list for c in c_names_indices ]) for c_ignore_list in to_ignore_trans_lists]):
                return
            self.trans_indices.append(c_names_indices)
            self.all_transformed.append( c_Ts[0].apply_transformation(*couple_image_gt ))
            return
        
        for c_trans_index in range(len(c_Ts)) :
            
            c_trans = c_Ts[c_trans_index]
            c_names_indices = names_idx+[TDO[c_trans.__name__]]
            if any([all([ c in c_names_indices for c in c_ignore_list ]) for c_ignore_list in to_ignore_trans_lists]):
                continue
            c_trans_couple = c_trans.apply_transformation(*couple_image_gt)
            self.trans_indices.append(c_names_indices)
            self.all_transformed.append( c_trans_couple )
            self.run_recurse(c_trans_couple, c_Ts[c_trans_index+1:], c_names_indices)
        return

            
if __name__ == "__main__":
    
    from skimage.io import imread
    
    im1_path = "/home/dell/Documents/LxGeo/repos/faultsDetectionDL/data/processed/partition_512_site_A_B/valid/image/Site_A_rot_45_204.tif"
    gt1_path = "/home/dell/Documents/LxGeo/repos/faultsDetectionDL/data/processed/partition_512_site_A_B/valid/gt/Site_A_rot_45_204.tif"
    
    im1 = imread(im1_path)
    gt1= imread(gt1_path)
    
    RT = recurse_transform((im1, gt1))
    RT.run_recurse(RT.image_couple, images_transformations_list, [])
    
    all_t = RT.get_all_transformed()
    
    
    
    
    