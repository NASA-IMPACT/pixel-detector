# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-04-02 12:13:51
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-05-16 10:52:03

from config import (
    PREDICT_THRESHOLD,
    OUTPUT_DIR,
)
from data_helper import (
    get_data,
)

import matplotlib
matplotlib.use('Agg')

from keras.models import load_model
from PIL import Image

import json
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Evaluate:

    def __init__(self, config):
        """
            config = {
                'type'          : < choice of 'pixel' or 'deconv'>,
                'num_neighbor'  : < n of the n*n matrix to take as
                                    input to the model>

                'jsonfile'      : <json containing ncfile, extent
                                    to subset ncfile and shapefile
                                    for training >

                'num_epoch'     : keras num_epoch

                'model_path'    : path to the keras model

                'batch_size'    : keras batch_size

                'eval_jrefacson'     : <json containing ncfile, extent
                                    to subset ncfile and shapefile
                                    for evaluation >

                'pred_json'     : <json containing ncfile and extent
                                    to subset for prediction>
            }
        """
        self.config = config
        self.predict_thres = PREDICT_THRESHOLD
        self.output_path = OUTPUT_DIR
        self.eval_img_path = 'evaluation_plot.png'
        self.band1_img_path = 'band1.png'
        self.band2_img_path = 'band2.png'
        self.band3_img_path = 'band3.png'
        self.pix_bmp_path = 'pred_pix_bitmap.bmp'
        self.t_bmp_path = 'true_bitmap.bmp'

    def evaluate(self):
        """
        Description: evaluation workflow
        """
        print('looking for model in', str(self.config['model_path']))
        self.model = load_model(str(self.config['model_path']))

        x_list, y_list, b_list, transforms = get_data(
            str(self.config['eval_json']),
            self.config['num_neighbor']
        )

        for _id, (x, y, b_list, transform) in \
                enumerate(zip(x_list, y_list, b_list, transforms)):

            # make output folder
            output_folder = os.path.join(self.output_path, str(_id))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # make prediction
            y_pred = self.model.predict(
                x, batch_size=self.config['batch_size'])
            y_pred = y
            y_pred = (y_pred > PREDICT_THRESHOLD) * 1.0
            y_mat = self.reshape_array_to_image(
                (y_pred) * 255.0,
                b_list.shape[0],
                b_list.shape[1])
            y_true = self.reshape_array_to_image(
                y * 255.0, b_list.shape[0], b_list.shape[1])

            # save predicted images
            self.save_image(y_mat, os.path.join(
                output_folder, self.pix_bmp_path))
            self.save_image(y_true, os.path.join(
                output_folder, self.t_bmp_path))

            # plot images and predictions
            self.plot_rgb(b_list[:, :, 1],
                          b_list[:, :, 2],
                          b_list[:, :, 0],
                          y_true,
                          y_mat,
                          output_folder).savefig(
                os.path.join(output_folder, self.eval_img_path))

    def reshape_array_to_image(self, dim1_array, x_shape, y_shape):
        """
        desc: reshape given 1D array to a 2D array of
                given x,y Dimensions
        """
        return np.asarray(dim1_array, dtype='uint8').reshape(
            (x_shape, y_shape), order='C')

    def plot_rgb(self, RL1, GL1, BL1, y_true, y_pred, output_folder):
        """Summary

        Args:
            RL1 (TYPE): Description
            GL1 (TYPE): Description
            BL1 (TYPE): Description
            y_true (numpy array): true feature
            y_pred (numpy array): predicted feature
            output_folder (string): folder to store plots

        Returns:
            TYPE: Description

        """
        print('b_list[1] shape', RL1)

        # normalization to true RGB
        GL1_true = 0.45 * ((RL1 / 25.5)**2 / 100) + 0.1 * \
            ((GL1 / 25.5)**2 / 100) + 0.45 * ((BL1 / 25.5)**2 / 100)
        GL1_true = np.maximum(GL1_true, 0)
        GL1_true = np.minimum(GL1_true, 1)
        RGBL1_veggie = np.dstack(
            [((RL1 / 25.5) ** 2 / 100),
             ((GL1 / 25.5) ** 2 / 100),
             ((BL1 / 25.5) ** 2 / 100)])

        # plotting RGB
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=250)
        axes[0, 0].imshow(RL1, cmap='Reds', vmax=255, vmin=0)
        axes[0, 0].set_title('Red', fontweight='semibold')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(GL1, cmap='Greens', vmax=255, vmin=0)
        axes[0, 1].set_title('Veggie', fontweight='semibold')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(BL1, cmap='Blues', vmax=255, vmin=0)
        axes[0, 2].set_title('Blue', fontweight='semibold')
        axes[0, 2].axis('off')
        plt.subplots_adjust(wspace=.02)
        axes[1, 0].imshow(RGBL1_veggie)
        axes[1, 0].axis('off')
        axes[1, 1].imshow(RGBL1_veggie)

        self.save_image(RL1, os.path.join(output_folder, 'r.bmp'))
        self.save_image(GL1, os.path.join(output_folder, 'g.bmp'))
        self.save_image(BL1, os.path.join(output_folder, 'b.bmp'))
        axes[1, 1].imshow(y_true, alpha=0.15,)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(y_pred)
        axes[1, 2].imshow(y_true, alpha=0.15,)
        axes[1, 2].axis('off')
        return plt

    def save_image(self, img_array, loc):
        """
        Desc    : save given 'img_array' as image in the given 'loc' location
        """
        Image.fromarray(img_array).convert('L').save(loc)


if __name__ == '__main__':

    ev = Evaluate(json.load(open('config.json')))
    ev.evaluate()

