import matplotlib
matplotlib.use('Agg')

from shape_utils import (
    bitmap_from_shp,
)
from config import (
    PREDICT_THRESHOLD,
    IMG_SCALE
)
from glob import glob
from data_preparer import PixelDataPreparer, PixelListPreparer
from keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize
from evaluate import Evaluate

import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import rasterio
import numpy.ma as ma
import numpy as np
import matplotlib.pyplot as plt


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class BatchEvaluate(Evaluate):

    def __init__(self, config, step_num=3):
        """init for Evaluate class
        """
        self.batch_size = config['batch_size']
        self.num_n = config['num_neighbor']
        self.model_path = config['model_path']
        self.val_dir = config['val_input_dir']
        self.save_dir = config['val_output_dir']

        self.model = load_model(self.model_path)
        print(self.model.summary())
        path_list = glob(self.val_dir + '/*.tif')

        paths_lists = [path_list[x:x + step_num] for x in range(
            0, len(path_list), step_num)]
        for paths in paths_lists:
            self.dataset = PixelListPreparer(paths,
                                             neighbour_pixels=self.num_n)
            self.dataset.iterate()
            self.evaluate()


if __name__ == '__main__':
    import json
    config = json.load(open('config.json'))
    ev = BatchEvaluate(config)
