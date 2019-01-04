# -*- coding: utf-8 -*-
"""Functions used to process the GeoTIFF satellite images."""

import rasterio
import rasterio.warp
import fiona
import os
import itertools
import numpy as np
import config
from config import TIFF_DIR as WGS84_DIR
import osgeo
from osgeo import gdal
from osgeo import osr
import numpy as np
import pdb
import subprocess
import cv2
from glob import glob
from PIL import Image
from shutil import copyfile

def iterate_through(nc_file_paths, output_file_name, expect_cache = True, create_cache=True):

    rast_mtx = list()

    for n_band, file in enumerate(nc_file_paths):
        nfile = 'NETCDF:"'+file[0]+'":Rad'
        if create_cache:
            nfile = 'NETCDF:"'+file+'":Rad'
            warp_options = gdal.WarpOptions(
                format = 'GTiff',
                outputType = gdal.GDT_Float32,
                resampleAlg = 5,
                outputBounds = extent,
                dstSRS = osr.SRS_WKT_WGS84
                )
            if not expect_cache:
                output_file_name = './test.tif'

            wr = gdal.Warp(output_file_name, nfile,options = warp_options)
            wr.FlushCache()

        try:
            rast = rasterio.open(os.path.join(output_file_name))
        except:
            print('cache not found, please verify')

        rast_mtx.append(rast.read(1))
        rast.close()
    return rast_mtx


def create_array_from_nc(ncFiles_path, fname, extent, res):
    '''
    Create Geotiff files from NC files and return their numpy array
    '''
    geotiff_paths = []

    # do NC -> geotiff -> WGS84 geotiff for each NC file
    n_band = 0
    cache_dir = os.path.join(WGS84_DIR,fname)
    create_cache = False

    if fname == '':
        print('Not expecting cache, Converting NC to WGS84 TIF Using gdal')
        rast_mtx = iterate_through(ncFiles_path, 'test.tif', create_cache = False, expect_cache = False)

    else:
        print('checking cache in: ',cache_dir)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            create_cache = True
            print('Expected cache but not found, Converting NC to WGS84 TIF Using gdal')

        rast_mtx = iterate_through(ncFiles_path, os.path.join(cache_dir,str(n_band)+'_WGS84'+'.tif'), create_cache = create_cache)

        print('shape of raster', np.moveaxis(np.asarray(rast_mtx),0,-1).shape)

    return np.moveaxis(np.asarray(rast_mtx),0,-1),os.path.join(cache_dir,str(1)+'_WGS84'+'.tif')

def histogram_equalize(img):
    return cv2.equalizeHist(img.astype('uint8'))


def band_list(loc,band_array,time):
    """
    Get ncfile paths matching the bands and time from the 'loc' location
    """
    path_list = []

    for band in band_array:
        print('fname:',loc,time)
        fname = glob(loc+'/*'+band+'*s'+time+'*.nc')
        if fname == []:
            print('fname null')
            return False
        else:

            path_list.append(fname)

    return path_list



