#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:25:27 2021

@author: Bilel Kanoun
"""
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import numpy as np
import geopandas as gpd
from shapely.geometry import box
#from scipy.ndimage.filters import gaussian_filter
from scipy.signal import spline_filter
import sys,os
import concurrent.futures
from rasterio import windows

def rasterize_from_profile(geometry_iter, c_profile, burn_value):
    """
    rasterize shapes of geometry iterator using the background profile and the burn value.
    returns a numpy array of raster
    """
    out_raster = rasterize(geometry_iter,
                     (c_profile["height"], c_profile["width"]),
                     fill=0,
                     transform=c_profile["transform"],
                     all_touched=True,
                     default_value=burn_value,
                     dtype=c_profile["dtype"])
    return out_raster

def apply_spline_filter(matrix, lmbda=5):
    """
    Apply spline filter on input matrix.
    Return ndArray
    """
    gaussian_applied= spline_filter(matrix, lmbda=lmbda) * 2
    gaussian_applied[ matrix>0 ]= matrix[matrix>0]
    return gaussian_applied

def rasterize_gdf(gdf, input_profile, burn_column="b_val"):
    """
    Inputs:
        gdf: geodataframe of geometries with burn value at column burn_column
        rio_dataset: rasterio dataset of refrence (background) image
        burn_column: column name in shapefile representing burning values
    """
            
    new_profile = input_profile
            
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

def rasterize_view(args):
    """
    Runs rasterization on a windowed view
    """
    geometry_view, shapes_gdf, profile, burn_column = args
    view_gdf = gpd.GeoDataFrame(geometry=[geometry_view], crs=shapes_gdf.crs)
    gdf_view = gpd.overlay(shapes_gdf, view_gdf, "intersection")
    if (len(gdf_view)>0):
        return rasterize_gdf(gdf_view, profile, burn_column)
    else:
        return np.zeros((profile["height"], profile["width"]), dtype=profile["dtype"])

def parralel_rasterization(shapes_gdf, output_dst, burn_column):
    """
    Split rasterization process
    Inputs:
        shapes_gdf: geodataframe of all features
        output_dst: output rio dataset to fill
        burn_column: str defining burn column in shapes_gdf
    """
    
    windows_list = [window for ij, window in output_dst.block_windows()]
    window_geometries = [ box(*windows.bounds(c_window,output_dst.transform)) for c_window in windows_list]
            
    # windows.transform(c_window,output_dst.transform)
    concurrent_args = []
    for c_window, c_window_geometrie in zip(windows_list, window_geometries):
        c_profile = output_dst.profile.copy()
        c_profile.update( 
            height=c_window.height,
            width=c_window.width,
            transform = windows.transform(c_window,output_dst.transform)
            )
        concurrent_args.append( (c_window_geometrie, shapes_gdf, c_profile, burn_column) )
    
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=8
        ) as executor:
        futures = executor.map(rasterize_view, concurrent_args)
        for window, result in zip(windows_list, futures):
            output_dst.write(result,1, window=window)
        

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
    
    input_profile = None
    with rasterio.open(reference_image) as input_dst:
        input_profile = input_dst.profile
    input_profile.update(nodata= 0)
    input_profile.update(driver="GTiff")
    input_profile.update(blockxsize=4096, blockysize=4096, tiled=True)
    input_profile.update(count=1)
        
    # load gdf
    shapes_gdf=gpd.read_file( shapefile_path )
    
    burn_column = "b_val"
    if len(sys.argv) == 5:
        burn_column=sys.argv[4]
        if burn_column not in shapes_gdf.columns:
            print("Missing burn column {} !".format(burn_column))
            print("Existing columns: {}".format(shapes_gdf.columns))
            sys.exit(1)
        else:
            print("Burn values count: {}".format(len(shapes_gdf[burn_column].unique())))
    
    if shapes_gdf.crs != input_profile["crs"]:
        shapes_gdf = shapes_gdf.to_crs(input_profile["crs"])
        
    #out_matrix = apply_spline_filter(out_matrix)
    
    with rasterio.open(output_path, mode="w", **input_profile) as output_dst:        
        #out_matrix=rasterize_gdf(shapes_gdf, input_profile, burn_column=burn_column)
        #output_dst.write(out_matrix,1)
        parralel_rasterization(shapes_gdf, output_dst, burn_column=burn_column)
    