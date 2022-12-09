#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:17:03 2022

@author: cherif
"""


import torch
from faultsDetectionDL.training.ptl.lightning_segmentation_model import lightningSegModel
from LxGeoPyLibs.dataset.patchified_dataset import CallableModel
from LxGeoPyLibs.dataset.raster_dataset import RasterDataset
from LxGeoPyLibs.geometry.rasterizers.polygons_rasterizer import polygons_to_multiclass

class temp_model(lightningSegModel, CallableModel):
    def __init__(self, **kwargs):
        CallableModel.__init__(self, bs=8)
        lightningSegModel.__init__(self, **kwargs)


if __name__ == "__main__":
    mdl_p = "./models/refactor/trial1/epoch=15-step=57839.ckpt"
    hparams_file = "./reports/refactor/trial1/benchmark-model/version_11/hparams.yaml"
    mdl = temp_model.load_from_checkpoint(mdl_p, hparams_file=hparams_file)
    mdl = mdl.cuda()
    
    in_raster_path = "./data/ALL_DATA/site_D8S1/images/clean_D8S1_Ortho.tif"
    use_3_bands_only = lambda x : x[:3,:,:]
    in_dataset = RasterDataset(in_raster_path, preprocessing=use_3_bands_only)
    
    softmax = lambda x: torch.nn.Softmax(dim=1)(x)
    from LxGeoPyLibs.vision.image_transformation import Trans_Identity, Trans_Rot90, Trans_Rot180, Trans_Rot270
    augs = None#[Trans_Identity(), Trans_Rot90(), Trans_Rot180(), Trans_Rot270()]
    in_dataset.predict_to_file("/home/cherif/Documents/LxGeo/temp_pred.tif", mdl, (512,512), post_processing_fn=softmax, augmentations=augs)
