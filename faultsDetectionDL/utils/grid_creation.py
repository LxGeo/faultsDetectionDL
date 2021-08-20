#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 12:28:04 2021

@author: Bilel Kanoun
"""
import sys

import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd

import math

from rasterio.mask import mask
import numpy as np

"""
Usage :
        python grid_creation.py RGB_file_path valid_shapefile_path output_shapefile_path tile_size 
"""    

def run_grid_creation(rgb_rio_dst, valid_geometry, tile_size):
    """
    Inputs:
        RGB rasterio dataset
        valid geometry (polygon or MultiPolygon)
        tile size
    Output:
        geodataframe of grids that contains geometry list and label list
    """
    #Define the dataset boundaries
    all_zone_geometry = box(*rgb_rio_dst.bounds)

    # Define the RGB image width and height
    size = rgb_rio_dst.shape
    IMG_HEIGHT = rgb_rio_dst.height
    IMG_WIDTH = rgb_rio_dst.width

    # Assert valid zone in RGB Dataset
    assert(valid_geometry.within(all_zone_geometry))

    # Define the grid blocks
    Nx_blocks = math.ceil(IMG_WIDTH/tile_size)
    Ny_blocks = math.ceil(IMG_HEIGHT/tile_size)

    grid_list=[]
    label_list=[]
    
    for ix in range(Nx_blocks):
        for iy in range(Ny_blocks):
            win = Window(ix*tile_size, iy*tile_size, tile_size, tile_size)
            
            c_box = box(*rasterio.windows.bounds(win,rgb_rio_dst.transform))
            
            if (c_box.intersects(valid_geometry)):
                grid_list.append(c_box)
                label_list.append(2)
                continue
            
            #out_array, _ = mask(rgb_rio_dst, [c_box],crop=True)
            out_array = rgb_rio_dst.read(window=win)[0:-1]
            
            out_array_summed=np.sum(out_array, axis=0)
            any_val=out_array_summed.flat[0]
            #if (np.all(out_array_summed==any_val)):
            if (len(np.unique(out_array_summed))>2):
                grid_list.append(c_box)
                label_list.append(0)
                continue
            
            grid_list.append(c_box)
            label_list.append(-1)
            
            out_gdf=gpd.GeoDataFrame({"geometry":grid_list, "TTV": label_list})
            out_gdf.crs=rgb_rio_dst.crs
    
    return out_gdf


if __name__ == "__main__":
    
    RGB_file = sys.argv[1]
    valid_shp_path = sys.argv[2]
    output_path = sys.argv[3]
    tile_size = 512
    
    
    if (len(sys.argv)>4):
        tile_size = sys.argv[4]

    # Define the tiles size
    tile_size = int(tile_size)
    
    # Read the valid shapefile
    valid_shp = gpd.read_file(valid_shp_path) 
    valid_geometry = valid_shp.geometry[0]
    
    # Read the RGB dataset
    with rasterio.open(RGB_file) as dataset:
        # Returns the created grid list and label list
        out_gdf = run_grid_creation(dataset, valid_geometry, tile_size)
        out_gdf.to_file(output_path)

