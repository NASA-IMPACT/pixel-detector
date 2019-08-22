import os
from PIL import Image
from glob import glob
import numpy as np
import rasterio


class PixelDataPreparer:

    def __init__(self, path, neighbour_pixels=4):
        self.path = path
        self.img_dims_list = list()
        self.shape_img = list()
        self.dataset = list()
        self.labels = list()
        self.neighbour_pixels = neighbour_pixels
        self.img_path_list = list()
        self.raster_transforms = list()

    def iterate(self):
        """iterate through list of images in directory
        """
        neighborhood = 2 * self.neighbour_pixels
        for file in glob(self.path + "/*.tif"):
            self.img_path_list.append(file)
            with rasterio.open(file) as rast:
                img = np.moveaxis(rast.read(), 0, -1)
                self.raster_transforms.append(rast.transform)
            width, height, channels = img.shape
            self.img_dims_list.append(img.shape)
            bitmap_file = file.replace('.tif', '.bmp')
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
            print(file)
            self.add_to_dataset(padded_img, labels)

    def add_to_dataset(self, image, labels):
        width, height = labels.shape
        number_of_pixels = 2 * self.neighbour_pixels
        for row in range(0, width):
            for column in range(0, height):
                self.dataset.append(
                    image[row:(row + number_of_pixels),
                          column:(column + number_of_pixels), :])
                label = 1 if labels[row, column] == 255 else 0
                self.labels.append(label)


if __name__ == '__main__':
    dp = PixelDataPreparer('../data/images/', neighbour_pixels=4)
    dp.iterate()
    print('done')


class PixelListPreparer:

    def __init__(self, paths_list, neighbour_pixels=4):
        self.paths_list = paths_list
        self.img_dims_list = list()
        self.shape_img = list()
        self.dataset = list()
        self.labels = list()
        self.neighbour_pixels = neighbour_pixels
        self.img_path_list = list()
        self.raster_transforms = list()

    def iterate(self):
        """iterate through list of images in directory
        """
        neighborhood = 2 * self.neighbour_pixels
        for file in self.paths_list:
            self.img_path_list.append(file)
            with rasterio.open(file) as rast:
                img = np.moveaxis(rast.read(), 0, -1)
                self.raster_transforms.append(rast.transform)
            width, height, channels = img.shape
            self.img_dims_list.append(img.shape)
            bitmap_file = file.replace('.tif', '.bmp')
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
            print(file)
            self.add_to_dataset(padded_img, labels)

    def add_to_dataset(self, image, labels):
        width, height = labels.shape
        number_of_pixels = 2 * self.neighbour_pixels
        for row in range(0, width):
            for column in range(0, height):
                self.dataset.append(
                    image[row:(row + number_of_pixels),
                          column:(column + number_of_pixels), :])
                label = 1 if labels[row, column] == 255 else 0
                self.labels.append(label)


if __name__ == '__main__':
    dp = PixelDataPreparer('../data/images/', neighbour_pixels=4)
    dp.iterate()
    print('done')
