#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:09:50 2021

@author: Bilel Kanoun
"""

import numpy as np
import rasterio as rio
import geopandas as gpd
from rasterio.plot import reshape_as_raster, reshape_as_image
from shapely.affinity import rotate as rotate_geometry
from skimage.transform import rotate as rotate_image
from shapely.geometry import box, Polygon
import tempfile
import affine
import click

from faultsDetectionDL.utils.grid_creation import run_grid_creation
from faultsDetectionDL.utils.tile_partition import run_partition
from faultsDetectionDL.utils.large_rasters_utils import rotate_large_raster

import sys,os


def run_balanced_partition(site_name, train_geometry, valid_geometry, rgb_rio_dst, gt_rio_dst, 
                           tile_size, output_folder, tile_type_column):
    
    x_size = rgb_rio_dst.transform[0]
    y_size = -rgb_rio_dst.transform[4]
    # Set the list of rotation angles
    angles_list = [0, 45, 90]
    
    train_rgb, out_transform = rio.mask.mask(rgb_rio_dst, [train_geometry], crop=True)
    train_rgb = reshape_as_image(train_rgb)
    train_gt, _ = rio.mask.mask(gt_rio_dst, [train_geometry], crop=True)
    train_gt = reshape_as_image(train_gt)
    
    for angle in angles_list:
        
        # Set current site name
        c_site_name = site_name + '_rot_{}'.format(angle)
        print( "Running partition for angle {} ".format(angle) )

        # Rotate RGB and gtreshape_as_image
        #rgb_rotated = rotate_image(reshape_as_image(rgb_rio_raster), angle, resize=True, preserve_range=True)
        #gt_rotated = rotate_image(reshape_as_image(gt_rio_raster), angle, resize=True, preserve_range=True)
        
        rgb_rotated = rotate_large_raster(train_rgb, angle)
        gt_rotated = rotate_large_raster(train_gt, angle)
        assert((rgb_rotated.shape[0] == gt_rotated.shape[0]) and (rgb_rotated.shape[1] == gt_rotated.shape[1]))
        
        # Creating new geotransformp for rotated rasters
        rgb_zone_geometry = box(*train_geometry.bounds)
        rotated_zone_envelop_bounds = rotate_geometry(rgb_zone_geometry, angle).bounds
        
        # Rotate train geometry
        rotated_train_geometry = rotate_geometry(train_geometry, angle, origin=rgb_zone_geometry.centroid)
        
        # Rotate valid geometry
        rotated_valid_geometry = rotate_geometry(valid_geometry, angle, origin=rgb_zone_geometry.centroid)
        
        new_geotransform = rio.transform.from_origin(
            rotated_zone_envelop_bounds[0],
            rotated_zone_envelop_bounds[-1],
            x_size, y_size)
        #new_geotransform= new_geotransform* affine.Affine.rotation(-angle, pivot=rgb_zone_geometry.centroid.coords[0])
        
        with tempfile.NamedTemporaryFile(delete=True) as rgb_rotated_tmpfile:
            with rio.open(rgb_rotated_tmpfile.name,
                               'w+', driver='GTiff',
                               height=rgb_rotated.shape[0],
                               width=rgb_rotated.shape[1],
                               count=rgb_rotated.shape[2],
                               dtype=rgb_rotated.dtype,
                               crs=rgb_rio_dst.crs,
                               transform=new_geotransform) as rgb_rotated_dst:
                
                with tempfile.NamedTemporaryFile(delete=True) as gt_rotated_tmpfile:
                    with rio.open(gt_rotated_tmpfile.name, 'w+', driver='GTiff',
                                  tiled=True, blockxsize=tile_size, blockysize=tile_size,
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
                        grid_gdf = run_grid_creation(rgb_rotated_dst, rotated_train_geometry, rotated_valid_geometry, tile_size)
                        
                        # tile partition
                        run_partition(c_site_name, grid_gdf, rgb_rotated_dst, gt_rotated_dst, 
                                      output_folder, tile_type_column)
                

@click.command()
@click.argument('site_name')
@click.argument('rgb_path', type=click.Path(exists=True))
@click.argument('gt_path', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path(file_okay=True))
@click.option('-vshp','--valid_shp_path', required=False, type=click.Path(exists=True))
@click.option('-tshp','--train_shp_path', required=False, type=click.Path(exists=True))
@click.option('-ts','--tile_size', type=click.IntRange(10, 2**15), required=True)
@click.option('-t_t_c', '--tile_type_column', type=str, default="TTV")
def main(site_name, rgb_path, gt_path, output_folder, valid_shp_path, train_shp_path, tile_size, tile_type_column):
    
    print("Running for site: {}".format(site_name))
    
    site_crs = None
    train_geometry = Polygon()
    with rio.open(rgb_path) as rgb_rio_dst:
        site_crs = rgb_rio_dst.crs
        train_geometry=box(*rgb_rio_dst.bounds)
        
    if train_shp_path:
        train_shp = gpd.read_file(train_shp_path)
        assert len(train_shp.geometry)==1, "Train zone shapefile contains 0 or more than one geometry!"
        if train_shp.crs != site_crs:
            train_shp = train_shp.to_crs(site_crs)
        train_geometry = train_shp.geometry[0]
    
    valid_geometry = Polygon()
    if valid_shp_path:
        valid_shp = gpd.read_file(valid_shp_path) 
        assert len(valid_shp.geometry)==1, "Valid zone shapefile contains 0 or more than one geometry!"
        if valid_shp.crs != site_crs:
            valid_shp = valid_shp.to_crs(site_crs)
        valid_geometry = valid_shp.geometry[0]
        
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    with rio.open(rgb_path) as rgb_rio_dst:
        with rio.open(gt_path) as gt_rio_dst:        
            run_balanced_partition(site_name, train_geometry, valid_geometry, rgb_rio_dst, gt_rio_dst, 
                               tile_size, output_folder, tile_type_column)
    
    save_file_path = os.path.join(output_folder, "sites_processed.txt")
    with open(save_file_path, 'a+') as file:
        file.write("{}\n".format(site_name))        
   

if __name__ == "__main__" :
    main()
    