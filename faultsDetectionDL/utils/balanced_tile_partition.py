#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:09:50 2021

@author: Bilel Kanoun
"""

import rasterio as rio
import geopandas as gpd
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.affinity import rotate as rotate_geometry
from skimage.transform import rotate as rotate_image
from shapely.geometry import box
import tempfile
import affine

from faultsDetectionDL.utils.grid_creation import run_grid_creation
from faultsDetectionDL.utils.tile_partition import run_partition

import sys,os


def run_balanced_partition(site_name, valid_geometry, rgb_rio_dst, gt_rio_dst, 
                           tile_size, output_folder, tile_type_column):
    
    x_size = rgb_rio_dst.transform[0]
    y_size = -rgb_rio_dst.transform[4]
    # Set the list of rotation angles
    angles_list = [0, 45, 90]
    
    for angle in angles_list:
        
        # Set current site name
        c_site_name = site_name + '_rot_{}'.format(angle)
        print( "Running partition for angle {} ".format(angle) )

        # Rotate RGB and gtreshape_as_image
        rgb_rotated = rotate_image(reshape_as_image(rgb_rio_dst.read()), angle, resize=True, preserve_range=True)
        gt_rotated = rotate_image(reshape_as_image(gt_rio_dst.read()), angle, resize=True, preserve_range=True)
        
        assert((rgb_rotated.shape[0] == gt_rotated.shape[0]) and (rgb_rotated.shape[1] == gt_rotated.shape[1]))
        
        # Creating new geotransformp for rotated rasters
        rgb_zone_geometry = box(*rgb_rio_dst.bounds)
        rotated_zone_envelop_bounds = rotate_geometry(rgb_zone_geometry, angle).bounds
        
        # Rotate valid geometry
        rotated_valid_geometry = rotate_geometry(valid_geometry, angle, origin=rgb_zone_geometry.centroid)
        
        new_geotransform = rio.transform.from_origin(
            rotated_zone_envelop_bounds[0],
            rotated_zone_envelop_bounds[-1],
            x_size, y_size)
        #new_geotransform= new_geotransform* affine.Affine.rotation(-angle, pivot=rgb_zone_geometry.centroid.coords[0])
        
        with tempfile.NamedTemporaryFile(delete=False) as rgb_rotated_tmpfile:
            with rio.open(rgb_rotated_tmpfile.name,
                               'w+', driver='GTiff',
                               height=rgb_rotated.shape[0],
                               width=rgb_rotated.shape[1],
                               count=rgb_rotated.shape[2],
                               dtype=rgb_rotated.dtype,
                               crs=rgb_rio_dst.crs,
                               transform=new_geotransform) as rgb_rotated_dst:
                
                with tempfile.NamedTemporaryFile(delete=False) as gt_rotated_tmpfile:
                    with rio.open(gt_rotated_tmpfile.name,
                                       'w+', driver='GTiff',
                                       height=gt_rotated.shape[0],
                                       width=gt_rotated.shape[1],
                                       count=1,
                                       dtype=rgb_rotated.dtype,
                                       crs=gt_rio_dst.crs,
                                       transform=new_geotransform) as gt_rotated_dst:
                        
                        # writing rgb array
                        rgb_rotated_dst.write(reshape_as_raster(rgb_rotated).astype(rgb_rotated_dst.meta["dtype"]))
                        
                        # writing gt array
                        gt_rotated_dst.write(reshape_as_raster(gt_rotated).astype(gt_rotated_dst.meta["dtype"]))
                        
                        # tile grid creation
                        grid_gdf = run_grid_creation(rgb_rotated_dst, rotated_valid_geometry, tile_size)
                        
                        # tile partition
                        run_partition(c_site_name, grid_gdf, rgb_rotated_dst, gt_rotated_dst, 
                                      output_folder, tile_type_column)
                



if __name__ == "__main__" :
    if len(sys.argv) <8:
        print("Missing arguments!")
        #print_help()
        sys.exit(1)
        
    site_name =sys.argv[1]
    valid_shp_path = sys.argv[2]
    rgb_path = sys.argv[3]
    gt_path = sys.argv[4]
    tile_size = sys.argv[5]
    tile_size = int(tile_size)
    output_folder = sys.argv[6]
    tile_type_column=sys.argv[7]
    
    valid_shp = gpd.read_file(valid_shp_path) 
    valid_geometry = valid_shp.geometry[0]
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    with rio.open(rgb_path) as rgb_rio_dst:
        with rio.open(gt_path) as gt_rio_dst:        
            run_balanced_partition(site_name, valid_geometry, rgb_rio_dst, gt_rio_dst, 
                               tile_size, output_folder, tile_type_column)
   