#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 18:28:02 2022

@author: cherif
"""

import click
import geopandas as gpd
import pandas as pd
import os


def create_faults_and_contour(in_gdf, p_buffer_val=6, m_buffer_val=1.5, resolution=0.0005):
    """
    Args:
        in_gdf: geodataframe of buildings polygons
        p_buffer_val: buffer value to apply positively on polygon
        m_buffer_val: buffer value to apply negatively on polygon
    Returns:
        out_gdf : geodataframe of buildings and contours with different b_col field value
    """
    
    gdf_p = in_gdf[["geometry"]].copy()
    gdf_m = in_gdf[["geometry"]].copy()
    gdf_p["geometry"] = in_gdf.buffer(p_buffer_val*resolution, join_style=0)
    gdf_m["geometry"] = in_gdf.buffer(m_buffer_val*resolution, join_style=0)
    
    contours_gdf = gpd.overlay(gdf_p, gdf_m, "difference")
    del gdf_p
    
    combined_gdf = gpd.GeoDataFrame( 
        pd.concat( [gdf_m[["geometry"]].assign(b_val=2), contours_gdf.assign(b_val=1) ] , ignore_index=True
                  ), crs=in_gdf.crs )
    
    return combined_gdf




@click.command()
@click.argument('input_shp_path', type=click.Path(exists=True))
@click.argument('output_shp_path', type=click.Path(exists=False))
@click.option('--resolution', type=click.FLOAT, default=0.0005)
def main(input_shp_path, output_shp_path, resolution):
    
    in_gdf = gpd.read_file(input_shp_path)
    out_gdf = create_faults_and_contour(in_gdf, resolution=resolution)
    out_shape_dir = os.path.dirname(output_shp_path)
    if not os.path.isdir(out_shape_dir) and out_shape_dir:
        os.makedirs(out_shape_dir)
    out_gdf.to_file(output_shp_path)

if __name__ == "__main__":
    main()