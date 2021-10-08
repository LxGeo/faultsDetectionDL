#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:15:27 2021

@author: cherif
"""
import os
from qgis.core import QgsRasterLayer
from qgis.core import QgsProject, QgsDateTimeRange, QgsRasterLayerTemporalProperties
import datetime
from PyQt5.QtCore import QFileInfo
multic_class_style_path="/home/cherif/Documents/multiclass_faults_style.qml"

def StringToRaster(raster):
    # Check if string is provided
    fileInfo = QFileInfo(raster)
    path = fileInfo.filePath()
    baseName = fileInfo.baseName()
    layer = QgsRasterLayer(path, baseName)
    QgsProject.instance().addMapLayer(layer)
    #layer.loadNamedStyle(multic_class_style_path)
    set_layer_no_data_value(layer, 0)

def set_layer_no_data_value(layer, val):
    layer.dataProvider().setNoDataValue(1, -1)

def set_fixed_temporal_range(layer, t_range):
    """
    Set fixed temporal range for raster layer
    :param layer: raster layer
    :param t_range: fixed temporal range
    """
    mode = QgsRasterLayerTemporalProperties.ModeFixedTemporalRange
    tprops = layer.temporalProperties()
    tprops.setIsActive(False)
    tprops.setIsActive(True)
    tprops.setMode(mode)
    tprops.setFixedTemporalRange(t_range)

def set_layers_ranges(layer_prefix):
    all_layers = [layer for layer in QgsProject.instance().mapLayers().values()]
    to_set_time_layers = list(filter(lambda x: x.name().startswith(layer_prefix), all_layers))
    sorted_to_set_time_layers = sorted(to_set_time_layers, key=lambda x: x.dataProvider().dataSourceUri())
    overall_start_date = datetime.datetime(2021, 1, 3)
    
    for l_idx, layer_to_set in enumerate(sorted_to_set_time_layers):
        c_start_day = overall_start_date + datetime.timedelta(days=l_idx)
        c_end_day = overall_start_date + datetime.timedelta(days=l_idx+1)
        t_range = QgsDateTimeRange(c_start_day, c_end_day)
        set_fixed_temporal_range(layer_to_set, t_range)
        

def load_epochs_preds(top_folder, string_filter="rot_0"):
    epochs_names = os.listdir(top_folder)
    epochs_names = sorted( epochs_names, key=lambda x: (x[0:2]))
    epochs_folders_paths = [os.path.join(top_folder, c_epoch_name) for c_epoch_name in epochs_names]
    for c_epoch_folder in epochs_folders_paths:
        c_folder_files = os.listdir(c_epoch_folder)
        chosen_image = list(filter(lambda x: string_filter in x, c_folder_files))[0]
        StringToRaster( os.path.join(c_epoch_folder, chosen_image) )
        
