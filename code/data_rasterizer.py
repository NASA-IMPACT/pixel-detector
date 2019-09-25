# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-07-02 15:33:11
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-09-25 12:31:57

from rasterio_utils import (
    wgs84_transform_memory,
    combine_rasters,
)

from glob import glob
from PIL import Image
from config import BANDS_LIST
from shape_utils import bitmap_from_shp
from pyorbital import astronomy

import numpy as np
import xarray
import json
import os
import sys


class DataRasterizer():

    def __init__(self, jsonfile, save_path, cza_correct):

        self.save_path = save_path
        self.parse_json(jsonfile)
        self.prepare_data(cza_correct)

    def prepare_data(self, cza_correct):

        for item in self.jsondict:
            ncpath = item['ncfile']
            nctime = item['nctime']
            nclist = self.list_bands(ncpath, BANDS_LIST, nctime)
            extent = item['extent']
            shapefile_path = item['shp']
            ext_str = 'time-{}-loc-{}_{}_{}_{}'.format(
                nctime, extent[0], extent[1], extent[2], extent[3])
            bmp_path = os.path.join(self.save_path, ext_str + '.bmp')
            tif_path = os.path.join(self.save_path, ext_str + '.tif')
            transform, res = self.rasterize_ncfiles(
                nclist, extent, tif_path, cza_correct,
            )
            bitmap_array = bitmap_from_shp(shapefile_path, transform, res)
            self.save_image(bitmap_array.astype('uint8') * 255, bmp_path)

    def rasterize_ncfiles(self, nclist, extent, img_path, cza_correct):

        ref_list = list()
        tf_list = list()
        for ncfile in nclist:

            ref, tf = self.rasterize_ncfile(
                ncfile, extent, cza_correct
            )

            ref_list.append(ref)
            tf_list.append(tf)
        assert len(set(tf_list)) == 1  # check if all transforms are equal
        combine_rasters(ref_list, tf_list[0], img_path)

        return tf, ref.shape

    def rasterize_ncfile(self, ncfile, extent, cza_correct):

        ds = xarray.open_dataset(str(ncfile), engine='h5netcdf')
        k = ds['kappa0'].data
        utc_time = ds['t'].data
        ds.close()
        mem_file = wgs84_transform_memory(ncfile, 'float32', extent)
        ref, transform = self.rad_to_ref(mem_file, k, utc_time, cza_correct)
        return ref, transform

    def rad_to_ref(self, mem_file, k, utc_time, cza_correct, gamma=2.0):

        with mem_file.open() as memfile:
            data_array = xarray.open_rasterio(memfile)
            rad = data_array[0].data * k
            transform = data_array.transform
            if cza_correct:
                x, y = np.meshgrid(data_array['x'], data_array['y'])
                cza = astronomy.cos_zen(utc_time, x, y)
                rad = rad * cza
            data_array.close()
            ref = np.clip(rad, 0, 1)
            ref_clipped = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)
            return ref_clipped.astype('uint8'), transform

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

    dp = DataRasterizer('../data/train_list2.json', '../data/images_fastcza/',
                        cza_correct=False)
