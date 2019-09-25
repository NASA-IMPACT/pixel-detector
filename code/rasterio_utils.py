# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-05-15 13:49:38
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-09-25 11:37:18

"""
Functions based on rasterio library: https://github.com/mapbox/rasterio
"""

from rasterio.warp import reproject, Resampling, aligned_target
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from pyproj import Proj

import numpy as np
import rasterio
import xarray
import fiona
import json
import math
import sys
import os


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

    ny = int(abs(lats[1] - lats[0]) * km_per_deg_at_eq / res)
    nx = int(abs(lons[1] - lons[0]) * km_per_deg_at_lat / res)

    return (nx, ny)


def rasterio_meta(src, extent, count, dtype):
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
    meta.update(count=count,
                driver='GTiff',
                crs={'init': 'epsg:4326'},
                transform=new_transform,
                width=width,
                height=height,
                nodata=0,
                dtype=dtype,
                )
    return meta


def generate_subsets(ncfile, center, cache_path, side_size):
    """ generate images given center and size of image, save them in filepath

    Args:
        center (TYPE): Description
        side_size (TYPE): Description
        file_path (TYPE): Description
    """
    temp_ncfile = f'NetCDF:{ncfile}:Rad'
    img_list = list()
    with rasterio.open(temp_ncfile, 'r') as src:
        geos_proj = Proj(src.crs.to_proj4())
        wgs_proj = Proj(init='EPSG:4326')
        center_xy = pyproj.transform(wgs_proj, geo_proj, *center)
        center_idx = src.index(*center_xy)
        corners = generate_corners(center_idx, side_size)
        for corner in corners:
            window = Window.from_slices((corner[0], corner[0] + side_size),
                                        (corner[1], corner[0] + side_size)
                                        )
            src_arr = src.read(window=window)
            with rasterio.open(cache_path, 'w',
                               driver='GTiff', width=side_size,
                               height=side_size, count=6,
                               ) as dest:
                dest.write(src_arr, file_path)

    def generate_corners(idx, side_size):

        side_half = side_size / 2
        corner_list = [idx[0] - side_half, idx[1] - side_half]
        corner_list += [idx[0] - side_half, idx[1]]
        corner_list += [idx[0], idx[1] - side_half]
        corner_list += [idx[0] + side_half, idx[1] + side_half]
        corner_list += [idx[0] + side_half, idx[1]]
        corner_list += [idx[0], idx[1] + side_half]

        corner_list += [idx[0] - side_size, idx[1] - side_size]
        corner_list += [idx[0] - side_size, idx[1]]
        corner_list += [idx[0], idx[1] - side_size]
        corner_list += [idx[0] + side_size, idx[1] + side_size]
        corner_list += [idx[0] + side_size, idx[1]]
        corner_list += [idx[0], idx[1] + side_size]

    return corner_list


def wgs84_group_transform(src_array, reference_ncfile, extent, save_path):

    temp_ncfile = f'NetCDF:{reference_ncfile}:Rad'
    dest_res = ()
    with rasterio.open(temp_ncfile, 'r') as src:
        dest_meta = rasterio_meta(src, extent, src_array.shape[2])
        with rasterio.open(save_path, 'w', **dest_meta) as dst:
            for i in range(1, src_array.shape[2] + 1):
                reproject(
                    source=np.flip(src_array[:, :, i - 1].astype('uint8'),
                                   axis=0),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dest_meta['transform'],
                    dst_crs=dest_meta['crs'],
                    resampling=Resampling.bilinear,
                )
        dest_res = dest_meta['height'], dest_meta['width']
    return dest_res, dest_meta['transform']


def wgs84_transform(ncfile, dtype, extent):
    """ returns array in wgs84 projection

    Args:
        src_array (numpy array): source array
        ncfile (string): BAND 1 path
        extent (list): subset extent

    Returns:
        TYPE: numpy array
    """
    save_path = ncfile.replace('.nc', '.tif')
    temp_ncfile = f'NetCDF:{ncfile}:Rad'
    with rasterio.open(temp_ncfile, 'r') as src:
        dest_meta = rasterio_meta(src, extent, 1, dtype)
        with rasterio.open(save_path, 'w', **dest_meta) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=dest_meta['transform'],
                      dst_crs=dest_meta['crs'],
                      resampling=Resampling.bilinear,
                      )
    return save_path


def wgs84_transform_memory(ncfile, dtype, extent):
    """ returns a memory file in wgs84 projection

    Args:
        src_array (numpy array): source array
        ncfile (string): BAND 1 path
        extent (list): subset extent

    Returns:
        TYPE: numpy array
    """
    save_path = ncfile.replace('.nc', '.tif')
    temp_ncfile = f'NetCDF:{ncfile}:Rad'
    memfile = MemoryFile()
    with rasterio.open(temp_ncfile, 'r') as src:
        dest_meta = rasterio_meta(src, extent, 1, dtype)
        with memfile.open(**dest_meta) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=src.transform,
                      src_crs=src.crs,
                      )
    return memfile


def read_tif(path, band=1):
    with rasterio.open(path, 'r') as ds:
        array = rasterio.band(ds, band)
        transform = ds.transform
    return array, transform


def combine_rasters(img_list, transform, save_path):

    meta = dict()
    meta['count'] = len(img_list)
    meta['driver'] = 'GTiff'
    meta['crs'] = {'init': 'epsg:4326'}
    meta['transform'] = transform
    meta['width'] = img_list[0].shape[0]
    meta['height'] = img_list[0].shape[1]
    meta['nodata'] = 0
    meta['dtype'] = 'uint8'

    with rasterio.open(save_path, 'w', **meta) as dest:
        for band_num, band in enumerate(img_list):
            dest.write(band, band_num + 1)
