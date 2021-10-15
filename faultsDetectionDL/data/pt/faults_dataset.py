#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:45:34 2021

@author: cherif
"""

import os
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_filtered_files(folder_to_search, include_extension=(".tif",) ):
    """
    """
    all_files = os.listdir(folder_to_search)
    filtered_files = filter( lambda x: any([c_ext.lower() in x.lower() for c_ext in include_extension]), all_files )
    filtered_files_full_path = map( lambda x: os.path.join(folder_to_search, x), filtered_files )
    return list(filtered_files_full_path)


class FaultsDataset(Dataset):
    
    def __init__(self, data_dir, augmentation_transforms=None,preprocessing=None, img_bands=3, num_classes=1 ,image_sub_dir="image", label_sub_dir="gt", include_extension=(".tif",)):
        
        self.img_dir = data_dir
        assert os.path.isdir(data_dir), "Can't find path {}".format(data_dir)
        self.image_dir = os.path.join(data_dir, image_sub_dir)
        self.label_dir = os.path.join(data_dir, label_sub_dir)
        
        self.images_paths = get_filtered_files(self.image_dir, include_extension=include_extension)
        self.labels_paths = get_filtered_files(self.label_dir, include_extension=include_extension)
        
        self.non_augmented_images_count = len(self.images_paths)
        assert self.non_augmented_images_count == len(self.labels_paths)
        
        self.augmentation_transforms = augmentation_transforms
        self.augmented_count = len(augmentation_transforms) * self.non_augmented_images_count
        
        self.img_bands=img_bands
        self.num_classes = num_classes
        #ENcoder related preprocessing
        self.preprocessing=preprocessing
        

    def __len__(self):
        return self.augmented_count
    
    def __getitem__(self, idx):
        
        image_idx = idx // (len(self.augmentation_transforms))
        transform_idx = idx % (len(self.augmentation_transforms))
        
        img_path = self.images_paths[image_idx]
        label_path = self.labels_paths[image_idx]
        
        img = imread(img_path)[:,:,0:self.img_bands]
        label = imread(label_path)
        
        c_trans = self.augmentation_transforms[transform_idx]
        img, label = c_trans.apply_transformation(img, label)
        
        if self.preprocessing:
            #img[:,:,0:3] = self.preprocessing(img[:,:,0:3])
            img = self.preprocessing(img)
        
        if self.num_classes==1:
            label = np.expand_dims(label, axis=-1)
            label = torch.from_numpy(label.copy()).permute(2, 0, 1).float()
        else:
            label = torch.from_numpy(label.copy()).long()
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        return img , label
        
        
        
        