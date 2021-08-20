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


images_transformations_list=[Trans_Rot90, Trans_Rot180, Trans_Rot270, Trans_Flipud, Trans_fliplr, Trans_gaussian_noise]