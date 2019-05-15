# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-05-15 13:49:38
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-05-15 14:52:39

"""
Functions based on rasterio library: https://github.com/mapbox/rasterio
"""

from rasterio.warp import reproject, Resampling, aligned_target
from rasterio.transform import Affine

import rasterio
import xarray
import fiona
import json
import math
import numpy as np
import os
import sys


def width_height(bbox, resolution_in_km=1.0):
    """ calculate width and height for the given bbox.

    Args:
        bbox (list): list of bbox coords (left, bottom, top, right)
        resolution_in_km (float, optional): size of a pixel (in KM)

    Returns:
        TYPE: Description
    """
    res = resolution_in_km

    lons = bbox[::2]
    lats = bbox[1::2]
    print(lats, lons)
    km_per_deg_at_eq = 111.
    km_per_deg_at_lat = km_per_deg_at_eq * np.cos(np.pi * np.mean(lats) / 180.)

    ny = int(abs(int((lats[1] - lats[0])) * km_per_deg_at_eq / res))
    nx = int(abs(int((lons[1] - lons[0])) * km_per_deg_at_lat / res))

    return (nx, ny)


def rasterio_meta(src, extent):
    """Form the meta for the new projection using source profile

    Args:
        src (rasterio object): source rasterio.Dataset object
        extent (list): list of boundary LatLon

    Returns:
        rasterio.Dataset.profile: modified meta file
    """
    meta = src.profile
    width, height = width_height(extent)
    new_transform = rasterio.transform.from_bounds(*extent, width, height)
    meta.update(count=1,
                driver='GTiff',
                crs={'init': 'epsg:4326'},
                transform=new_transform,
                width=width,
                height=height,
                nodata=0,)
    return meta


def wgs84_transform(src_array, ncfile, extent):
    """ returns array in wgs84 projection

    Args:
        src_array (numpy array): source array
        ncfile (string): BAND 1 path
        extent (list): subset extent

    Returns:
        TYPE: numpy array
    """
    temp_ncfile = f'NetCDF:{ncfile}:Rad'
    with rasterio.open(temp_ncfile, 'r') as src:
        dest_meta = rasterio_meta(src, extent)
        dest_array = np.zeros((dest_meta['width'], dest_meta['height']))
        reproject(np.flip(src_array, axis=0),
                  dest_array,
                  src_transform=src.transform,
                  src_crs=src.crs,
                  dst_transform=dest_meta['transform'],
                  dst_crs=dest_meta['crs'],
                  resampling=Resampling.nearest,
                  )
        return dest_array, dest_meta['transform']
