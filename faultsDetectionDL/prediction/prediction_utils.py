#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:04:35 2021

@author: cherif
"""


import os, sys
import rasterio as rio
import click
import geopandas as gpd
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.affinity import rotate as rotate_geometry
from skimage.transform import rotate as rotate_image
from shapely.geometry import box
from faultsDetectionDL.prediction.model_loader import ModelLoader
from faultsDetectionDL.prediction.prediction_strategies import PatchOverlapStrategy
import tempfile
import affine


def run_multi_prediction(site_rio_dst, output_folder, rot_angles, checkpoints_paths, prediction_strategy, custom_objects_config,
                         patch_size, num_bands):
    """
    Runs prediction for site using a set of 
    """
    
    ML = ModelLoader(custom_objects_config)
    # define common variables
    x_size = site_rio_dst.transform[0]
    y_size = -site_rio_dst.transform[4]
    site_zone_geometry = box(*site_rio_dst.bounds)
    site_crs= site_rio_dst.crs
    
    # Initiating angle raster dict
    ang_raster_map = { 0:site_rio_dst }
    
    for c_angle in rot_angles:
        if (c_angle==0):
            continue
        print("Creating rotation raster for angle {}".format(c_angle))
        
        rotated_array = rotate_image(reshape_as_image(site_rio_dst.read()), c_angle, resize=True, preserve_range=True)
        rotated_zone_envelop_bounds = rotate_geometry(site_zone_geometry, c_angle).bounds
        new_geotransform = rio.transform.from_origin(
            rotated_zone_envelop_bounds[0],
            rotated_zone_envelop_bounds[-1],
            x_size, y_size)
        with tempfile.NamedTemporaryFile(delete=False) as c_rotated_raster_tmpfile:
            c_rotated_raster_dst = rio.open(c_rotated_raster_tmpfile.name,
                               'w+', driver='GTiff',
                               height=rotated_array.shape[0],
                               width=rotated_array.shape[1],
                               count=rotated_array.shape[2],
                               dtype=rotated_array.dtype,
                               crs=site_crs,
                               transform=new_geotransform)
            c_rotated_raster_dst.write(reshape_as_raster(rotated_array).astype(c_rotated_raster_dst.meta["dtype"]))
            ang_raster_map[c_angle] = c_rotated_raster_dst
    
    for c_checkpoint_path in checkpoints_paths:
        checkpoint_name = os.path.basename(c_checkpoint_path)
        print("Prediction for checkpoint: {}".format(checkpoint_name))
        c_checkpoint_output_folder = os.path.join(output_folder, checkpoint_name)
        if not (os.path.isdir(c_checkpoint_output_folder)):
            os.makedirs(c_checkpoint_output_folder)
        c_loaded_model = ML.get_model(c_checkpoint_path)
        for c_angle in ang_raster_map: 
            c_rotated_raster_dst = ang_raster_map[c_angle]
            pred_startegy = PatchOverlapStrategy(c_loaded_model, c_rotated_raster_dst, patch_size, num_bands)
            pred_startegy.predict()
        
        
        
        
        


@click.command()
@click.argument('site_raster_path', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path(file_okay=False))
@click.option('-ra','--rot_angles', type=click.IntRange(0, 180), multiple=True, required=True, help="Rotation angles to be applied (degrees)")
@click.option('-cp', '--checkpoints_paths', type=click.Path(exists=True), required=True, multiple=True, help="Models checkpoints paths to load.")
@click.option('-ps', '--patch_size', type=int, required=True, help="Patch size used at training phase")
@click.option('-nb', '--num_bands', type=int, required=True, help="Number of bands used while training phase")
def main(site_raster_path, output_folder, rot_angles, checkpoints_paths, patch_size, num_bands):
    """
    """
    if (0 not in rot_angles):
        print("Addding angle 0 to rot_angles")
        rot_angles = rot_angles + (0,)
    
    if( len(set(rot_angles)) != len(rot_angles)):
        print("Some rotation angles are duplicated!")
        print("Exiting multiprediction!")
        return
    
    if( len(set(checkpoints_paths)) != len(checkpoints_paths)):
        print("Some checkpoints paths are duplicated!")
        print("Exiting multiprediction!")
        return
    
    if not(os.path.isdir(output_folder)):
        os.makedirs(output_folder)
    
    with rio.open(site_raster_path) as site_rio_dst:
        run_multi_prediction(site_rio_dst, output_folder, rot_angles, checkpoints_paths, custom_config, patch_size, num_bands)


if __name__ == "__main__":
    main()
    
