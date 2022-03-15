#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:19:47 2022

@author: cherif
"""


import click
import geopandas as gpd
import rasterio as rio
import rasterio.fill
from rasterio.plot import reshape_as_raster, reshape_as_image
import os
import tempfile
from skimage import measure
from osgeo import gdal
import numpy as np
import alphashape
from shapely.affinity import affine_transform
from faultsDetectionDL.utils.multi_value_rasterization import rasterize_from_profile 
import shutil

def affine2tuple(aff):
    return (aff.a, aff.b, aff.d, aff.e, aff.xoff, aff.yoff)

def get_valid_extents_mask(img, invalid_val):
    valid_contour_pts=[]
    R,C = img.shape
    out_mask = np.zeros_like(img,np.uint8)
    
    for c_row in range(R):
        st_col = np.argmax(img[c_row,:])
        end_col = np.argmax(img[c_row,::-1])
        #if (st_col!=0): valid_contour_pts.append([c_row,st_col])
        #if (end_col!=0): valid_contour_pts.append([c_row,C-end_col])
        #if (st_col!=0 or end_col!=0): 
        out_mask[c_row, st_col:C-end_col]+=1
    
    for c_col in range(C):
        st_row = np.argmax(img[:,c_col])
        end_row = np.argmax(img[::-1,c_col])
        #if (st_row!=0): valid_contour_pts.append([st_row,c_col])
        #if (end_row!=0): valid_contour_pts.append([R-end_row,c_col])
        #if (st_row!=0 or end_row!=0): 
        out_mask[st_row:R-end_row, c_col]+=1
    return out_mask.astype(bool)

def get_valid_extents_pts(img, invalid_val):
    """"""
    valid_contour_pts=[]
    R,C = img.shape
    
    for c_row in range(R):
        st_col = np.argmax(img[c_row,:])
        end_col = np.argmax(img[c_row,::-1])
        valid_contour_pts.append([c_row,st_col])
        valid_contour_pts.append([c_row,C-end_col])
    
    for c_col in range(C):
        st_row = np.argmax(img[:,c_col])
        end_row = np.argmax(img[::-1,c_col])
        valid_contour_pts.append([st_row,c_col])
        valid_contour_pts.append([R-end_row,c_col])
    return valid_contour_pts


@click.command()
@click.argument('input_image_path', type=click.Path(exists=True))
@click.argument('output_image_path', type=click.Path(exists=False))
@click.option('--max_search_radius_pixel', type=click.INT, default=100)
@click.option('--nan_value', type=click.FLOAT, default=255)
@click.option('--ref_band', type=click.INT, default=0)
def main(input_image_path, output_image_path, max_search_radius_pixel, nan_value, ref_band):
    """
    """
    
    input_image=None
    input_profile=None
    with rio.open(input_image_path) as in_dst:
        input_image = in_dst.read()
        input_profile = in_dst.profile.copy()
    
    inner_mask_profile = input_profile.copy(); inner_mask_profile.update(count=1, dtype=rio.uint8)
    
    accross_bands_mask=np.sum([ input_image[b_i,:,:] == input_profile["nodata"] for b_i in range(input_image.shape[0])], axis=0 ).astype(bool)
    mask_img = input_image[ref_band,:,:] == nan_value
    mask_img = np.logical_or(accross_bands_mask, mask_img)
    inv_mask_img = ~mask_img # represents binary of non null pixels
    
    # turn all nan to zeros
    input_image[:,mask_img] = 0
    with rio.open(output_image_path, "w", **input_profile) as out_dst:
        out_dst.write(input_image)
    """
    contours = measure.find_contours(inv_mask_img,0.9)    
    contours_pts = np.vstack(contours)
    """
    optim_alpha = 1/max_search_radius_pixel#alphashape.optimizealpha(contours_pts)
    
    #inner_image_mask = get_valid_extents_mask(inv_mask_img, 0)
    contours_pts = get_valid_extents_pts(inv_mask_img, 0)
    contours_pts = list(map(lambda t:(t[1],t[0]), contours_pts))
    out_shape_pixel_coords = alphashape.alphashape(contours_pts[:], optim_alpha)
    
    out_shape = affine_transform(out_shape_pixel_coords, affine2tuple(input_profile["transform"]))
    
    inner_image_mask = rasterize_from_profile([out_shape], inner_mask_profile, 1)
    
    to_interpolate_mask = ~np.logical_and(mask_img, inner_image_mask)
         
    
    #to_interpolate_mask = inv_mask_img
    with tempfile.NamedTemporaryFile(delete=False) as mask_tmpfile:
        with rio.open(mask_tmpfile.name, "w", **inner_mask_profile) as mask_dst:
            mask_dst.write(np.expand_dims(to_interpolate_mask, 0))
    
    
    ET = gdal.Open(output_image_path, gdal.GA_Update)    
    MT = gdal.Open(mask_tmpfile.name, gdal.GA_ReadOnly)
    mask_band = MT.GetRasterBand(1)
    
    for band_idx in range(input_image.shape[0]):
        ETband = ET.GetRasterBand(band_idx+1)
        gdal.FillNodata(targetBand = ETband, maskBand = mask_band, 
                     maxSearchDist = max_search_radius_pixel, smoothingIterations = 0)        
    


if __name__ == "__main__":
    main()