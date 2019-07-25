# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-07-02 15:33:11
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-07-25 12:59:54

from glob import glob
from PIL import Image
from config import BANDS_LIST
from rasterio_utils import wgs84_group_transform
from shape_utils import bitmap_from_shp

import json
import numpy as np
import os
import sys
import scipy
import xarray


class DataPreparer():

    def __init__(self, jsonfile, save_path):

        self.save_path = save_path
        self.parse_json(jsonfile)
        self.prepare_data()

    def prepare_data(self):

        for item in self.jsondict:
            ncpath = item["ncfile"]
            nctime = item["nctime"]
            nclist = self.list_bands(ncpath, BANDS_LIST, nctime)
            extent = item["extent"]
            shapefile_path = item["shp"]
            ext_str = "_{}_{}_{}_{}_{}".format(
                extent[0], extent[1], extent[2], extent[3], nctime)
            save_str = os.path.join(self.save_path, ext_str)
            img_path = os.path.join(self.save_path, save_str, '.tif')
            bmp_path = os.path.join(self.save_path, save_str, '.bmp')
            import pdb;pdb.set_trace()
            res, transform = self.rasterize_ncfiles(nclist, extent,
                                                    shapefile_path, img_path)
            bitmap_array = bitmap_from_shp(shapefile_path, transform, res)
            self.save_image(bitmap_array, bmp_path)

    def rasterize_ncfiles(self, nclist, extent, img_path):

        ref_list = list()
        for i, ncfile in enumerate(nclist):
            with xarray.open_dataset(str(ncfile), engine='h5netcdf') as ds:
                k = ds['kappa0'].data
                rad = ds['Rad'].data
                ref = np.clip(rad * k, 0, 1)
                if 'RadF' in ds.dataset_name:
                    res = (10848, 10848)
                else:
                    res = (5000, 3000)
                gamma = 2.0
                if i == 3 or i == 5:  # if BAND_4 or BAND_6, then upsample
                    ref = scipy.ndimage.zoom(ref, 2, order=0)
                if i == 1:  # if BAND_2 then downsample
                    ref = self.rebin(ref, [res[0], res[1]])
                ref_255 = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)
                ref_list.append(ref_255)
        ref_stack = np.dstack(ref_255)
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
        shape = shape[0], arr.shape[0] // shape[0], shape[1], arr.shape[1] // shape[1]
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
            Image.fromarray(img_array).convert('L').save(loc)
        if ndims == 3 and img_array.shape[2] == 3:
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


class UnetDataPreparer(DataPreparer):

    def __init__(self, jsonfile, save_path, image_size=256):

        self.image_size = image_size
        self.save_path = save_path
        self.jsonfile = jsonfile

    def prepare_tiles(self, subset_ratio=1.2):

        jsonfile = self.jsonfile
        datasets = get_data(jsonfile)
        img_idx = 0
        for idx, data in enumerate(zip(*datasets)):
            print(data[1])
            _, label, bands, _ = data
            width, height = bands.shape[0:2]
            bitmap = self.reshape_array_to_image(
                label, width, height)
            width_ratio = width / self.image_size
            height_ratio = height / self.image_size
            if width_ratio > subset_ratio and height_ratio > subset_ratio:
                top_left_points = [
                    [0, 0],
                    [width - self.image_size, 0],
                    [0, height - self.image_size],
                    [width - self.image_size, height - self.image_size]
                ]
                for point in top_left_points:
                    loc = os.path.join(self.save_path, str(img_idx))
                    print('saving in', loc)
                    self.save_image(bitmap[point[0]:point[0] + self.image_size,
                                           point[1]:point[1] + self.image_size,
                                           ]*255,
                                    loc + '_label.bmp')
                    self.save_image(
                        bands[point[0]:point[0] + self.image_size,
                              point[1]:point[1] + self.image_size,
                              0:3,
                              ],
                        loc + '_rgb.png')
                    img_idx += 1

    def get_unet_data(self):

        rgb_images = []
        bitmap_images = []
        for file in glob(self.save_path + '/*.png'):
            bitmap_file = file[:-7] + 'label.bmp'
            try:
                rgb_images.append(np.array(Image.open(file)))
                bitmap_images.append(np.array(Image.open(bitmap_file)))
            except FileNotFoundError:
                print('file not found')

        return np.array(rgb_images), np.array(bitmap_images)


if __name__ == '__main__':

    dp = DataPreparer('../data/train_list2.json', '../data/images/')
