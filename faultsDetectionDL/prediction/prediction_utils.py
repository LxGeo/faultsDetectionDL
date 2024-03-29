#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:04:35 2021

@author: cherif
"""


import os, sys
import rasterio as rio
import numpy as np
import click
import geopandas as gpd
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.affinity import rotate as rotate_geometry
from skimage.transform import rotate as rotate_image
from shapely.geometry import box
from faultsDetectionDL.utils.large_rasters_utils import rotate_large_raster

from faultsDetectionDL.prediction.prediction_strategies import PatchOverlapStrategy, Basic2Strategy
import tempfile
import affine
from faultsDetectionDL.prediction.model_loader import GenericModel,get_model_wrapper

framework_options = GenericModel._available_frameworks

import warnings
warnings.filterwarnings("ignore")


def inv_rotate_raster( to_rotate_back_dst, reference_dst, angle, output_path ):
    """
    Run reverse rotation on a raster based on refrence dst
    """
    #refrence_central_pixel_position = reference_dst.height//2, reference_dst.width//2
    
    inv_rotated_array = rotate_large_raster(reshape_as_image(to_rotate_back_dst.read()), -angle)
    #inv_rotated_array = np.round(inv_rotated_array)
    inv_rotated_central_pixel_position=inv_rotated_array.shape[0]//2, inv_rotated_array.shape[1]//2
    
    output_profile = to_rotate_back_dst.profile.copy()
    output_profile.update(tiled=True, blockxsize=256, blockysize=256,
                          height=reference_dst.height, width=reference_dst.width, transform=reference_dst.transform)
    with rio.open(output_path,mode="w+", **output_profile) as out_dst:
        out_dst.write( reshape_as_raster(
            inv_rotated_array[ 
                inv_rotated_central_pixel_position[0]-reference_dst.height//2:inv_rotated_central_pixel_position[0]+reference_dst.height//2,
                inv_rotated_central_pixel_position[1]-reference_dst.width//2:inv_rotated_central_pixel_position[1]+reference_dst.width//2,
                :
                ].astype(out_dst.meta["dtype"])
            ) )

def preds_to_labels(in_preds_path, out_labels_path):
    """
    """
    with rio.open(in_preds_path) as in_dst:
        out_profile = in_dst.profile
        out_profile["count"]=1
        out_profile["dtype"]=rio.uint8
        out_profile["nodata"]=None
        
        with rio.open(out_labels_path, "w", **out_profile) as out_dst:
            out_dst.write( np.expand_dims(np.argmax(in_dst.read(),0),0) )
    
def run_multi_prediction(site_rio_dst, output_folder, rot_angles, checkpoints_paths, prediction_strategy,
                         patch_size, framework_model_wrapper):
    """
    Runs prediction for site using a set of 
    """
    
    # define common variables
    x_size = site_rio_dst.transform[0]
    y_size = -site_rio_dst.transform[4]
    site_zone_geometry = box(*site_rio_dst.bounds)
    site_crs= site_rio_dst.crs
    
    # Initiating angle raster dict
    ang_raster_map = { 0:site_rio_dst }
    
    for c_angle in rot_angles:
        """if (c_angle==0):
            continue"""
        print("Creating rotation raster for angle {}".format(c_angle))
        
        rotated_array = rotate_large_raster(reshape_as_image(site_rio_dst.read()), c_angle)
        rotated_zone_envelop_bounds = rotate_geometry(site_zone_geometry, c_angle).bounds
        new_geotransform = rio.transform.from_origin(
            rotated_zone_envelop_bounds[0],
            rotated_zone_envelop_bounds[-1],
            x_size, y_size)
        with tempfile.NamedTemporaryFile(delete=True) as c_rotated_raster_tmpfile:
            c_rotated_raster_dst = rio.open(c_rotated_raster_tmpfile.name,
                               'w+', driver='GTiff',
                               tiled=True,
                               blockxsize=patch_size,
                               blockysize=patch_size,
                               height=rotated_array.shape[0],
                               width=rotated_array.shape[1],
                               count=rotated_array.shape[2],
                               dtype=rotated_array.dtype,
                               crs=site_crs,
                               transform=new_geotransform,
                               nodata=site_rio_dst.nodata)
            c_rotated_raster_dst.write(reshape_as_raster(rotated_array).astype(c_rotated_raster_dst.meta["dtype"]))
            c_rotated_raster_dst.close()
            c_rotated_raster_dst = rio.open(c_rotated_raster_tmpfile.name, "r")
            ang_raster_map[c_angle] = c_rotated_raster_dst
            
            
    for c_checkpoint_path in checkpoints_paths:
        checkpoint_name = os.path.basename(c_checkpoint_path)
        print("Prediction for checkpoint: {}".format(checkpoint_name))
        c_checkpoint_output_folder = os.path.join(output_folder, checkpoint_name)
        if not (os.path.isdir(c_checkpoint_output_folder)):
            os.makedirs(c_checkpoint_output_folder)
        c_loaded_model = framework_model_wrapper(c_checkpoint_path)
        for c_angle in ang_raster_map: 
            c_rotated_raster_dst = ang_raster_map[c_angle]
            pred_startegy = prediction_strategy(c_loaded_model, c_rotated_raster_dst, patch_size=patch_size, n_classes=c_loaded_model.n_classes)
            pred_startegy.fill_out_dataset()
            # saving c_angle back rotation image in current checkpoint folder
            c_back_rot_path = os.path.join(c_checkpoint_output_folder, "pred_rot_{}.tif".format(c_angle))
            print("Applying reverse rotation!")
            inv_rotate_raster( pred_startegy.out_rio_dst, ang_raster_map[0], c_angle, c_back_rot_path )
            
            c_back_rot_labels_path = os.path.join(c_checkpoint_output_folder, "labels_rot_{}.tif".format(c_angle))
            preds_to_labels(c_back_rot_path,c_back_rot_labels_path)
            pass
        
        
        
        
        


@click.command()
@click.argument('site_raster_path', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path(file_okay=False))
@click.option('-ra','--rot_angles', type=click.IntRange(0, 359), multiple=True, required=True, help="Rotation angles to be applied (degrees)")
@click.option('-cp', '--checkpoints_paths', type=click.Path(exists=True), required=True, multiple=True, help="Models checkpoints paths to load.")
@click.option('-ps', '--patch_size', type=int, required=True, help="Patch size used for prediction phase")
@click.option('-fn', '--framework_name', required=True, default=None,type=click.Choice(framework_options, case_sensitive=False), help="Framework used for predicition")
def main(site_raster_path, output_folder, rot_angles, checkpoints_paths, patch_size, framework_name):
    """
    """
    if (0 not in rot_angles):
        print("Addding angle 0 to rot_angles")
        #rot_angles = rot_angles + (0,)
    
    if( len(set(rot_angles)) != len(rot_angles)):
        print("Some rotation angles are duplicated!")
        print("Exiting multiprediction!")
        return
    
    if( len(set(checkpoints_paths)) != len(checkpoints_paths)):
        print("Some checkpoints paths are duplicated!")
        print("Exiting multiprediction!")
        return
    
    model_extension_dict ={ "tf":"hdf5", "pt": ".pth", "ptl": ".ckpt"}
    model_extension=model_extension_dict.get(framework_name.lower(), None)
    
    if len(checkpoints_paths)==1 and os.path.isdir(checkpoints_paths[0]):
        list_of_files_in_dir = os.listdir(checkpoints_paths[0])
        list_of_files_in_dir = [ os.path.join(checkpoints_paths[0], c_file) for c_file in list_of_files_in_dir ]
        checkpoints_paths = list(filter(lambda x: x.endswith(model_extension), list_of_files_in_dir ))
    
    if not(os.path.isdir(output_folder)):
        os.makedirs(output_folder)    
    
    framework_model_wrapper = get_model_wrapper(framework_name)
    
    with rio.open(site_raster_path) as site_rio_dst:
        run_multi_prediction(site_rio_dst, output_folder, rot_angles, checkpoints_paths, Basic2Strategy,
                             patch_size, framework_model_wrapper)


if __name__ == "__main__":
    main()
    
