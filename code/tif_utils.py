# -*- coding: utf-8 -*-
"""Functions used to process the GeoTIFF satellite images."""

import cv2
import numpy as np
import os
import rasterio
import rasterio.warp
import sys

from rasterio import Affine
from rasterio.transform import xy
from config import TIFF_DIR as WGS84_DIR
from osgeo import gdal
from osgeo import osr
from glob import glob


def iterate_through(nc_file_paths, cache_folder_name, extent, res, expect_cache=True, create_cache=True):
    '''
    Iterate through the NCfile paths are convert to geotiff if create_cache=True
    else read geotiffs from cache_folder
    Returns: (res[0],res[1],[<num of bands>]) numpy array of geotiffs in cache folder
    '''

    rast_mtx = list()

    for n_band, file in enumerate(nc_file_paths):
        nfile = 'NETCDF:"' + file[0] + '":Rad'

        output_file_name = os.path.join(cache_folder_name, str(n_band) + '_WGS84.tif')

        raster_transform = ''
        if create_cache:

            warp_options = gdal.WarpOptions(
                format      ='GTiff',
                outputType  =gdal.GDT_Float32,
                resampleAlg =5,
                width       =res[0],
                height      =res[1],
                outputBounds=extent,
                dstSRS      =osr.SRS_WKT_WGS84
            )
            wr = gdal.Warp(output_file_name, nfile, options=warp_options)
            wr.FlushCache()

            rast = rasterio.open(output_file_name)
            rast_mtx.append(histogram_equalize(rast.read(1)))
            rast.close()

        if not expect_cache:
            # this flow is when there should not be any intermediate
            # file stored. all conversions happens in-memory

            output_file_name = './test.tif'
            nfile = 'NETCDF:"' + file[0] + '":Rad'

            warp_options = gdal.WarpOptions(
                format      ='VRT',
                outputType  =gdal.GDT_Float32,
                resampleAlg =5,
                width       =res[0],
                height      =res[1],
                outputBounds=extent,
                dstSRS      =osr.SRS_WKT_WGS84
            )
            wr = gdal.Warp('', nfile, options=warp_options)
            wr.FlushCache()

            if not raster_transform:
                # based on https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html

                c,a,b,f,d,e = wr.GetGeoTransform()
                raster_transform = Affine(a,b,c,d,e,f)

            rast_mtx.append(histogram_equalize(wr.ReadAsArray()))


    return rast_mtx, raster_transform


def create_array_from_nc(ncFiles_path, fname, extent, res):
    '''
    Create Geotiff files from NC files and return their numpy array
    '''

    geotiff_paths = []

    # do NC -> geotiff -> WGS84 geotiff for each NC file
    cache_dir = os.path.join(WGS84_DIR, fname)

    if fname == '':
        print('Not expecting cache, Converting NC to WGS84 TIF Using gdal')

        rast_mtx,raster_transform = iterate_through(ncFiles_path,
            create_cache=False,
            expect_cache=False
            )

        return np.moveaxis(np.asarray(rast_mtx), 0, -1), raster_transform

    else:
        print('checking cache in: ', cache_dir)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            create_cache = True
            print('Expected cache but not found, Converting NC to WGS84 TIF Using gdal')

        else:
            print('Cache Found, Using Cache...')

        rast_mtx, raster_transform = iterate_through(ncFiles_path,
            cache_dir,
            extent, res,
            create_cache=create_cache
            )

        print('shape of raster', np.moveaxis(np.asarray(rast_mtx), 0, -1).shape)

        return np.moveaxis(np.asarray(rast_mtx), 0, -1), cache_dir


def histogram_equalize(img):
    #return img
    return cv2.equalizeHist(img.astype('uint8'))


def band_list(loc, band_array, time):
    """
    Get ncfile paths matching the bands and time from the 'loc' location
    """
    path_list = []

    for band in band_array:
        print('fname:', loc, time)
        fname = glob(loc + '/*' + band + '*s' + time + '*.nc')
        if fname == []:
            print('Nc Files not Found')
            return False
        else:

            path_list.append(fname)

    return path_list
