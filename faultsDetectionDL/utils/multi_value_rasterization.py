#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:25:27 2021

@author: Bilel Kanoun
"""
import rasterio
from rasterio.features import rasterize
import numpy as np
import geopandas as gpd
from scipy.ndimage.filters import gaussian_filter
import sys,os

def rasterize_from_profile(geometry_iter, profile, burn_value):
    """
    rasterize shapes of geometry iterator using the background profile and the burn value.
    returns a numpy array of raster
    """
    return rasterize(geometry_iter,
                     (profile["height"], profile["width"]),
                     fill=0,
                     transform=profile["transform"],
                     all_touched=True,
                     default_value=burn_value,
                     dtype=profile["dtype"])

def apply_gaussian(matrix, sigma=0.9):
    """
    Apply gaussian filter on input matrix.
    Return ndArray
    """
    gaussian_applied= gaussian_filter(matrix, sigma)
    gaussian_applied[ matrix>0 ]= matrix[matrix>0]
    return gaussian_applied

def rasterize_gdf(gdf, input_profile, burn_column="b_col"):
    """
    Inputs:
        gdf: geodataframe of geometries with burn value at column burn_column
        rio_dataset: rasterio dataset of refrence (background) image
        burn_column: column name in shapefile representing burning values
    """
            
    new_profile = input_profile
        
    new_profile.update(
        dtype=rasterio.float32,
        count=1
        )
    
    rasterization_images=[]
    
    if (not (burn_column in gdf.columns)):
        single_rasterization= rasterize_from_profile(gdf.geometry, new_profile, 1)
        rasterization_images.append(single_rasterization)
        
    else:
        burning_values=np.sort(gdf[burn_column].unique())
        for c_burn_value in burning_values:
            c_burn_gdf_view = gdf[gdf[burn_column]==c_burn_value]
            single_rasterization = rasterize_from_profile(c_burn_gdf_view.geometry, new_profile, c_burn_value)
            rasterization_images.append(single_rasterization)
       
    return np.max(rasterization_images, axis=0)

def print_help():
    
    help_message="""
        python multi_value_rasterization.py shapefile_path reference_image output_path *[burn_column]
    """
    print(help_message)
    
if __name__ == "__main__":
    
    if len(sys.argv) <4:
        print("Missing arguments!")
        print_help()
        sys.exit(1)
    
    shapefile_path =sys.argv[1]
    reference_image = sys.argv[2]
    output_path = sys.argv[3]
    
    # load gdf
    shapes_gdf=gpd.read_file( shapefile_path )
    
    burn_column = "b_col"
    if len(sys.argv) == 5:
        burn_column=sys.argv[4]
        if burn_column not in shapes_gdf.columns:
            print("Missing burn column {} !".format(burn_column))
            print("Existing columns: {}".format(shapes_gdf.columns))
            sys.exit(1)
        else:
            print("Burn values count: {}".format(len(shapes_gdf[burn_column].unique())))
    
    input_profile = None
    with rasterio.open(reference_image) as input_dst:
        input_profile = input_dst.profile
    input_profile.update(nodata= -32768)
    
    out_matrix=rasterize_gdf(shapes_gdf, input_profile, burn_column=burn_column)
    
    #out_matrix = apply_gaussian(out_matrix)
    
    with rasterio.open(output_path, mode="w", **input_profile) as output_dst:
        output_dst.write(out_matrix,1)
    