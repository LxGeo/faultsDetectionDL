#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:54:53 2021

@author: dell
"""
import sys, os

import rasterio as rio
from rasterio import mask
from image_transformation import images_transformations_list
from rasterio.plot import reshape_as_raster, reshape_as_image
import geopandas as gpd
from tqdm import tqdm

import numpy as np


#import math

def run_partition(site_name, tiles_gdf, rgb_rio_dst, gt_rio_dst, output_folder, 
                  tile_type_column="TTV"):
    
    for tile_type_name in ["valid", "train"]:
        for img_type_name in ["image", "gt"]:
            folder_to_create = os.path.join(output_folder, tile_type_name, img_type_name)
            if not os.path.isdir(folder_to_create):
                os.makedirs(folder_to_create)
    
    # Variable_defining current tile transformation count
    c_tile_transformation_count = 1
    
    sample_tile = tiles_gdf.geometry[0]
    (minx, miny, maxx, maxy) = sample_tile.bounds
    
    # Get pixelSizeX, pixelSizeY    
    pixelSizeX, pixelSizeY = rgb_rio_dst.res
    
    # Get the crs
    rgb_crs = rgb_rio_dst.crs
    
    # Get tile_sizex and tile_sizey
    tile_sizex = round((maxx-minx)/pixelSizeX)
    tile_sizey = round((maxy-miny)/pixelSizeY)
    
    def save_transform(ID, geotransform, rgb_matrix, gt_matrix, tile_type):
        
        output_path_templ = output_folder+"/{0}/{{}}/{{}}_{{}}.tif".format(tile_type)
        with rio.open(output_path_templ.format('image',site_name, ID),
                      'w', driver="GTiff" , width=tile_sizex, height=tile_sizey, count=4, crs=rgb_crs, transform=geotransform, nodata=-32768, dtype=rio.float32) as output_rgb_dst:
            output_rgb_dst.write(reshape_as_raster(rgb_matrix).astype(np.float32))
            
        with rio.open(output_path_templ.format('gt',site_name, ID),
                      'w', driver="GTiff" , width=tile_sizex, height=tile_sizey, count=1, crs=rgb_crs, transform=geotransform, nodata=-32768, dtype=rio.float32) as output_gt_dst:
            output_gt_dst.write(reshape_as_raster(gt_matrix).astype(np.float32))
    
    for c_row in tqdm(tiles_gdf.iterrows(), desc="Iterating grids", total=len(tiles_gdf)):

        c_TTV = c_row[1]["TTV"]
        if (c_TTV==-1):
            continue
        
        assert (c_TTV in [0,2])
        
        c_box = c_row[1]["geometry"]
        rgb_image, rgb_transform = rio.mask.mask(rgb_rio_dst,[c_box], crop=True, nodata=-32768)
        gt_image, gt_transform = rio.mask.mask(gt_rio_dst,[c_box], crop=True, nodata=-32768)
        
        rgb_image = reshape_as_image(rgb_image)
        gt_image = reshape_as_image(gt_image)
                
        tile_type = "valid" if c_TTV==2 else "train"
        
        save_transform(c_tile_transformation_count, rgb_transform, rgb_image, gt_image, tile_type)
        c_tile_transformation_count+=1

            
def print_help():
    
    help_message="""
        python tile_partition.py site_name tiles_shp_path rgb_path gt_path output_folder *[tile_type_column] *[transform_count_column] 
        Default: *tile_type_column="TTV"
    """
    print(help_message)
    
if __name__=="__main__":
    
    if len(sys.argv) <6:
        print("Missing arguments!")
        print_help()
        sys.exit(1)
        
    site_name =sys.argv[1]
    tiles_shp_path = sys.argv[2]
    rgb_path = sys.argv[3]
    gt_path = sys.argv[4]
    output_folder = sys.argv[5]
    
    # load gdf
    tiles_gdf=gpd.read_file( tiles_shp_path )
    
    tile_type_column="TTV"
    
    if len(sys.argv) == 7:
        tile_type_column=sys.argv[6]
        if tile_type_column not in tiles_gdf.columns:
            print("Missing tile type column {} !".format(tile_type_column))
            print("Existing columns: {}".format(tiles_gdf.columns))
            sys.exit(1)
                
    with rio.open(rgb_path,'r') as rgb_rio_dst:
        with rio.open(gt_path,'r') as gt_rio_dst:
            run_partition(site_name, tiles_gdf, rgb_rio_dst, gt_rio_dst, output_folder, 
                  tile_type_column)
