import cv2
import numpy as np
import rasterio
import tensorflow as tf

from glob import glob


class UnetGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, data_path,
                 to_fit=True, batch_size=8, dim=(256, 256),
                 n_channels=6, shuffle=True):
        'Initialization'
        self.data_path = data_path
        self.tif_list = [filename for filename in glob(f'{data_path}*.tif')]
        self.mask_list = []

        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n = 0
        self.max = len(self)
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.tif_list) / self.batch_size))

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.tif_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[
            index * self.batch_size: (index + 1) * self.batch_size
        ]

        # Find list of IDs
        tif_list_temp = [self.tif_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(tif_list_temp)

        if self.to_fit:
            y = self._generate_y(tif_list_temp)
            return X, y
        else:
            return X

    def _generate_X(self, tif_list_temp):
        '''Generates data containing batch_size images'''
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(tif_list_temp):
            # Store sample
            X[i, ] = _load_tif_image(ID, self.dim)

        return X

    def _generate_y(self, tif_list_temp):
        '''Generates data containing batch_size masks'''
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)

        # Generate data
        for i, ID in enumerate(tif_list_temp):
            # Store sample
            y[i, ] = _load_grayscale_image(ID.replace(
                '.tif', '.bmp'), self.dim
            )

        return y

    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def _load_grayscale_image(image_path, dim):
    '''Load grayscale image'''
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    return np.expand_dims(cv2.resize(img, dim), -1)


def _load_tif_image(image_path, dim):
    '''load tif image'''

    with rasterio.open(image_path, 'r') as data:
        return cv2.resize(
            np.moveaxis(data.read(), 0, -1), dim
        )
