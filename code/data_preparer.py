# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-07-02 15:33:11
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-07-03 15:29:03

import numpy as np
import json
from data_helper import get_data
from PIL import Image
import os
from glob import glob

class UnetDataPreparer:

    def __init__(self, jsonfile, save_path, image_size=256):

        self.image_size = image_size
        self.save_path = save_path
        self.jsonfile = jsonfile


    def parse_json(self, jsonfile):
        self.jsondict = {}
        with open(jsonfile) as js:
            self.jsondict = json.loads(js.read())

    def reshape_array_to_image(self, dim1_array, x_shape, y_shape):
        """
        desc: reshape given 1D array to a 2D array of
                given x,y Dimensions
        """
        return np.asarray(dim1_array, dtype='uint8').reshape(
            (x_shape, y_shape), order='C')

    def convert_to_true_rgb(self, img):

        RL1, GL1, BL1 = img[:,:,1], img[:,:,2], img[:,:,0]
        GL1_true = 0.45 * ((RL1 / 25.5)**2 / 100) + 0.1 * \
            ((GL1 / 25.5)**2 / 100) + 0.45 * ((BL1 / 25.5)**2 / 100)
        GL1_true = np.maximum(GL1_true, 0)
        GL1_true = np.minimum(GL1_true, 1)

        RGBL1_true = np.dstack([((RL1 / 25.5) ** 2 / 100),
                                 GL1_true,
                                 ((BL1 / 25.5) ** 2 / 100)])
        return RGBL1_true*255

    def save_image(self, img_array, loc):
        """
        Desc    : save given 'img_array' as image in the given 'loc' location
        """

        ndims = len(img_array.shape)
        if ndims == 2:
            Image.fromarray(img_array).convert('L').save(loc)
        if ndims == 3 and img_array.shape[2] == 3:
            Image.fromarray(img_array.astype('uint8')).convert('RGB').save(loc)

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

    dp = UnetDataPreparer(jsonfile='../data/train_list.json',save_path='../data/unet_images')
    dp.prepare_tiles()
