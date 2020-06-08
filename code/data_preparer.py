import numpy as np
import os
import rasterio
import time

from glob import glob
from PIL import Image


class PixelListPreparer:

    def __init__(self, img_path_list, neighbour_pixels=4):
        self.img_path_list = img_path_list
        self.img_dims_list = list()
        self.shape_img = list()
        self.dataset = list()
        self.labels = list()
        self.neighbour_pixels = neighbour_pixels
        self.raster_transforms = list()

    def iterate(self, bands):
        """iterate through list of images in directory
        """

        neighborhood = 2 * self.neighbour_pixels
        for image_path in self.img_path_list:
            with rasterio.open(image_path) as rast:
                img = np.moveaxis(rast.read(), 0, -1)
                self.raster_transforms.append(rast.transform)
            width, height, channels = img.shape
            self.img_dims_list.append(img.shape)
            bitmap_file = image_path.replace('.tif', '.bmp')
            if os.path.exists(bitmap_file):
                labels = np.array(Image.open(bitmap_file))
            else:
                labels = np.zeros((width, height))
            padded_img = np.zeros(
                (
                    width + neighborhood,
                    height + neighborhood,
                    channels
                ), dtype='uint8')
            padded_img[self.neighbour_pixels:-self.neighbour_pixels,
                       self.neighbour_pixels:-self.neighbour_pixels, :] = img
            self.add_to_dataset(padded_img, labels, bands)

    def add_to_dataset(self, image, labels, bands):
        """Loops through a single image and creates a matched dataset of image segments and labels
        from a particular set of bands.

        Args:
            image (np.array(dtype='unit8')): Numpy array resulting from the conversion of a GeoTIFF
                by PixelListPreparer.iterate()

            labels (np.array(dtype='unit8')): Binary array of the same dimensions as the input image
                where 1 represents a smoke pixel and 0 represents non-smoke pixel

            bands (list): List containing integer band labels. This will be used to determine which
                bands are added to self.dataset.

        Returns:
            No direct return, but modifies self.dataset and self.labels. self.dataset will be
            a list of neighbour_pixels*2 x neighbour_pixels*2 np.arrays with a corresponding self.
            labels that contains the label for smoke or nonsmoke
        """

        height, width = labels.shape
        number_of_pixels = 2 * self.neighbour_pixels
        for row in range(0, height):
            for column in range(0, width):
                self.dataset.append(
                    image[row:(row + number_of_pixels),
                          column:(column + number_of_pixels), bands])
                label = 1 if labels[row, column] == 255 else 0
                self.labels.append(label)


class PixelDataPreparer(PixelListPreparer):

    def __init__(self, path, neighbour_pixels=4):
        paths = glob(path + "/*.tif")
        super().__init__(paths, neighbour_pixels=neighbour_pixels)


if __name__ == '__main__':
    dp = PixelDataPreparer('../data/images/', neighbour_pixels=4)
    dp.iterate([0, 1, 2, 3, 4, 5])
    print('done')
