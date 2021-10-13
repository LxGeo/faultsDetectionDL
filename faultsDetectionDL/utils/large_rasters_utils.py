# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:54:37 2021

@author: geoimage
"""

import os
import pyvips
import numpy as np

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def rotate_large_raster(raster_descriptor, rotation_angle):
    """
    Args:
        raster_descriptor could be:
            str -> path of raster to rotate
            ndarray -> numpy array
        rotation_angle: double
    Returns:
        numpy array
    """
    vips_image = None
    if type(raster_descriptor) == str :
        assert os.path.exists(raster_descriptor), "Raster not found!"
        vips_image = pyvips.Image.new_from_file(raster_descriptor)
    elif type(raster_descriptor) == np.ndarray:
        vips_image = numpy2vips(raster_descriptor)
    else:
        raise Exception("Not recognized raster descriptor of type {}".format(type(raster_descriptor)))
    
    rot_vips_image = vips_image.rotate(rotation_angle)    
    return vips2numpy(rot_vips_image)
