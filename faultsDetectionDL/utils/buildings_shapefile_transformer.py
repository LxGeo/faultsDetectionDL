# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:45:00 2021

@author: geoimage
"""

import click
import geopandas as gpd
import pandas as pd


def add_buildings_edges(in_gdf, p_buffer_val=0.7, m_buffer_val=-0.1):
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
    gdf_p["geometry"] = in_gdf.buffer(p_buffer_val, join_style=2)
    gdf_m["geometry"] = in_gdf.buffer(m_buffer_val, join_style=2)
    
    contours_gdf = gpd.overlay(gdf_p, gdf_m, "difference")
    del gdf_m, gdf_p
    
    combined_gdf = gpd.GeoDataFrame( 
        pd.concat( [in_gdf[["geometry"]].assign(b_val=1), contours_gdf.assign(b_val=2) ] , ignore_index=True
                  ), crs=in_gdf.crs )
    
    return combined_gdf
    


@click.command()
@click.argument('input_shp_path', type=click.Path(exists=True))
@click.argument('output_shp_path', type=click.Path(exists=False))
def main(input_shp_path, output_shp_path):
    
    in_gdf = gpd.read_file(input_shp_path)
    out_gdf = add_buildings_edges(in_gdf)
    out_gdf.to_file(output_shp_path)

if __name__ == "__main__":
    main()