import matplotlib
matplotlib.use('Agg')

from data_preparer import PixelDataPreparer
from keras.models import load_model
from PIL import Image
from shape_utils import (
    bitmap_from_shp,
)
from config import (
    PREDICT_THRESHOLD,
    IMG_SCALE,

)

import os
import rasterio
import numpy as np
import pandas as pd
import seaborn as sn
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Evaluate:

    def __init__(self, config, num_n):
        """init for Evaluate class
        """
        self.batch_size = config['batch_size']
        self.num_n = config['num_neighbor']
        self.model_path = config['model_path']
        self.val_dir = config['val_dir']

        self.dataset = PixelDataPreparer(self.val_dir,
                                         neighbour_pixels=self.num_n
                                         )
        self.dataset.iterate()
        self.model = load_model(self.model_path)
        print(self.model.summary())
        self.save_dir = config['val_output_dir']
        self.evaluate

    def evaluate(self):
        """
        evaluate workflow
        """
        input_data = self.dataset.dataset
        labels = self.dataset.labels
        predicted_pixels = self.get_predictions(input_data, labels)
        self.visualize_predictions(self.dataset, predicted_pixels)

    def get_predictions(self, input_data, labels):
        """return predictions for given input data
        Args:
            input_data (TYPE): input data
        Returns:
            TYPE: prediction array list
        """
        last_idx = 0
        prediction_list = []
        conf_mtx = np.zeros((2, 2))
        total_images = len(self.dataset.img_dims_list)
        for i, img_shape in enumerate(self.dataset.img_dims_list):
            print('predicting {} of {} images:'.format(
                i + 1, total_images))
            next_idx = last_idx + img_shape[0] * img_shape[1]
            img_prediction = self.__predict__(
                input_data[last_idx: next_idx])
            img_true = labels[last_idx: next_idx]
            conf_mtx += confusion_matrix(img_true,
                                         img_prediction > PREDICT_THRESHOLD,)
            last_idx = next_idx
            prediction_list.append(img_prediction.reshape(
                (img_shape[0], img_shape[1])))
        return prediction_list

    def __predict__(self, data):
        """give out predictions for the given image
        Returns:
            TYPE: Description
        """
        pred_bmp = self.model.predict(
            np.array(data), batch_size=self.batch_size
        )
        return pred_bmp

    def convert_rgb(self, img_path):
        with rasterio.open(img_path) as rast:
            red = rast.read(2)
            blue = rast.read(1)
            pseudo_green = rast.read(3)
            height, width = red.shape
            img = np.moveaxis(
                np.array([red, pseudo_green, blue]), 0, -1
            )
        return img

    def plot_figures(self, img, prediction, tif_name):

            width, height = img[0].shape
            fig = plt.figure()
            fig.set_size_inches(width / height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(self.save_dir,
                                     tif_name.replace('.tif', '.png')),
                        dpi=height)
            thres_pred = np.asarray(prediction * IMG_SCALE,
                                    dtype='uint8')

            thres_pred_masked = ma.masked_where(
                thres_pred <= PREDICT_THRESHOLD * IMG_SCALE, thres_pred)

            plt.imshow(thres_pred_masked, alpha=0.35, cmap='spring')
            plt.axis('off')

            plt.savefig(os.path.join(self.save_dir,
                                     tif_name.replace('.tif', '_masked.png')),
                        dpi=height)

    def visualize_predictions(self, dataset, predictions, ):
        """plot  prediction heatmaps along with original image
        for evaluation purposes
        Args:
            x_path (TYPE): list of image paths
            predictions (TYPE): numpy arrays of predictions
        """
        x_path = dataset.img_path_list

        assert len(x_path) == len(predictions)
        for i, prediction in enumerate(predictions):
            img_name = os.path.basename(x_path[i])
            print('plotting {} of {} images:'.format(i + 1, len(predictions)))

            img = self.convert_rgb(x_path[i])
            self.plot_figures(img, prediction, img_name)


if __name__ == '__main__':
    import json
    config = json.load(open('config.json'))
    ev = Evaluate(config, num_n=7)
