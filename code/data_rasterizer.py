# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-07-02 15:33:11
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-09-17 15:38:35

from config import (
    SAT_H,
    SAT_LON,
    SAT_SWEEP
)
from glob import glob
from PIL import Image
from config import BANDS_LIST
from rasterio_utils import wgs84_group_transform
from shape_utils import bitmap_from_shp
from pyproj import Proj
from pyorbital import astronomy

import json
import numpy as np
import os
import sys
import scipy.ndimage
import xarray


class DataRasterizer():

    def __init__(self, jsonfile, save_path, cza_correct):

        self.save_path = save_path
        self.parse_json(jsonfile)
        self.prepare_data(cza_correct)

    def prepare_data(self, cza_correct):

        for item in self.jsondict:
            ncpath = item["ncfile"]
            nctime = item["nctime"]
            nclist = self.list_bands(ncpath, BANDS_LIST, nctime)
            extent = item["extent"]
            shapefile_path = item["shp"]
            ext_str = "time-{}-loc-{}_{}_{}_{}".format(
                nctime, extent[0], extent[1], extent[2], extent[3])
            img_path = os.path.join(self.save_path, ext_str + '.tif')
            bmp_path = os.path.join(self.save_path, ext_str + '.bmp')
            res, transform = self.rasterize_ncfiles(
                nclist, extent, img_path, cza_correct,
            )
            bitmap_array = bitmap_from_shp(shapefile_path, transform, res)
            self.save_image(bitmap_array.astype('uint8') * 255, bmp_path)

    def rasterize_ncfiles(self, nclist, extent, img_path, cza_correct):

        ref_list = list()
        for i, ncfile in enumerate(nclist):
            with xarray.open_dataset(str(ncfile), engine='h5netcdf') as ds:
                k = ds['kappa0'].data
                rad = ds['Rad'].data
                rad = rad * k
                if cza_correct:
                    geos_proj = Proj(proj='geos', h=SAT_H,
                                     lon_0=SAT_LON, sweep=SAT_SWEEP)
                    x = ds['x'].data * SAT_H
                    y = ds['y'].data * SAT_H
                    utc_time = ds['t'].data
                    x_mesh, y_mesh = np.meshgrid(x, y)
                    lons, lats = geos_proj(x_mesh, y_mesh, inverse=True)
                    cza = np.zeros((rad.shape[0], rad.shape[1]))
                    cza = astronomy.cos_zen(utc_time, lons, lats)
                    rad = rad * cza

                ref = np.clip(rad, 0, 1)
                if 'RadF' in ds.dataset_name:
                    res = (10848, 10848)
                else:
                    res = (3000, 5000)
                gamma = 2.0
                if i == 3 or i == 5:  # if BAND_4 or BAND_6, then upsample
                    ref = scipy.ndimage.zoom(ref, 2, order=0)
                if i == 1:  # if BAND_2 then downsample
                    ref = self.rebin(ref, [res[0], res[1]])
                ref_255 = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)
                ref_list.append(ref_255)
        ref_stack = np.dstack(ref_list)
        res, transform = wgs84_group_transform(ref_stack, nclist[0],
                                               extent, img_path
                                               )
        return res, transform

    def rebin(self, arr, shape):
        """ rebins band 2 to 1KM resolution

        Args:
            a (TYPE): array to rebin
            shape (TYPE): Description

        Returns:
            TYPE: Description
        """
        shape = shape[0], arr.shape[0] // shape[0], shape[1], \
        arr.shape[1] // shape[1]
        return arr.reshape(shape).mean(-1).mean(1)

    def list_bands(self, loc, band_array, time):
        """
        Get ncfile paths matching the bands and time from the 'loc' location
        """

        path_list = []

        print('checking for nc in ', loc)
        for band in band_array:
            fname = glob('{}/*{}*s{}*.nc'.format(loc, band, time))
            print('{}/*{}*s{}*.nc'.format(loc, band, time))
            print(fname)
            if fname == []:
                print('Nc Files not Found')
                sys.exit(1)
            else:
                path_list += fname

        return path_list

    def parse_json(self, jsonfile):
        self.jsondict = {}
        with open(jsonfile) as js:
            self.jsondict = json.loads(js.read())

    def true_rgb(self, img):
        """convert reflectance to True RGB color range

        Args:
            img (TYPE): Description

        Returns:
            TYPE: Description
        """
        red, green, blue = img[:, :, 1], img[:, :, 2], img[:, :, 0]
        green_true = 0.45 * ((red / 25.5)**2 / 100) + 0.1 * \
            ((green / 25.5)**2 / 100) + 0.45 * ((blue / 25.5)**2 / 100)
        green_true = np.maximum(green_true, 0)
        green_true = np.minimum(green_true, 1)

        rgb_true = np.dstack([red,
                              green_true,
                              blue]
                             )
        return rgb_true

    def save_image(self, img_array, loc):
        """
        Desc    : save given 'img_array' as image in the given 'loc' location

        Args:
            img_array (numpy Array): array to convert to image
            loc (String): save path
        """

        ndims = len(img_array.shape)
        if ndims == 2:
            Image.fromarray(img_array).save(loc)
        elif ndims == 3 and img_array.shape[2] == 3:
            Image.fromarray(img_array.astype('uint8')).convert('RGB').save(loc)
        else:
            print('unsupported array dimensions')

    def reshape_array_to_image(self, dim1_array, x_shape, y_shape):
        """
        desc: reshape given 1D array to a 2D array of
                given x,y Dimensions
        """
        return np.asarray(dim1_array, dtype='uint8').reshape(
            (x_shape, y_shape), order='C')


if __name__ == '__main__':

    dp = DataRasterizer('../data/eval_list.json', '../data/images_val_no_cza/',
                      cza_correct=False)
