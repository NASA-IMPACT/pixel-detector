# @author Muthukumaran R.

import matplotlib
matplotlib.use('Agg')

import cv2
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio


from keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix

from config import (
    PREDICT_THRESHOLD,
    OUTPUT_DIR,
    BAND_1_FILENAME,
    EVAL_DISP_STR
)
from preprocessing import (
    get_arrays_from_json,
    convert_bmp_to_shp,
    get_bitmap_from_shp,
    IOU_score
)


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
        self.band_img_path = 'band1.png'
        self.pix_bmp_path = 'pred_pix_bitmap.bmp'
        self.shp_bmp_path = 'pred_shp_bitmap.bmp'
        self.t_bmp_path = 'true_bitmap.bmp'
        self.shp_path = 'shapefile.shp'
        self.true_overlay = 'true_overlay.png'
        #self.model_holder = MODELS[self.config['type']] (self.config)

    def evaluate(self):
        """
        desc:   produce images containing predictions along with their scores
                scores produced: Accuracy, precision, recall, IOU score.
        """

        print('looking for model in', str(self.config['model_path']))
        self.model = load_model(str(self.config['model_path']))

        x_list, y_list, cache_list = get_arrays_from_json(self.config['eval_json'],
                                                          self.config['num_neighbor']
                                                          )
        for x, y, cache_dir in zip(x_list, y_list, cache_list):
            # iterate through each x,y pair

            output_folder = os.path.join(self.output_path,
                                         os.path.basename(os.path.normpath(cache_dir)))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            self.make_predict_bitmaps(x, output_folder, cache_dir)

            # save BAND 1 raster from cache for evaluation purposes
            b1_rast = Image.open(os.path.join(
                cache_dir, BAND_1_FILENAME)).convert('RGBA')
            b1_rast.save(os.path.join(output_folder, self.band_img_path))

            self.plot_evaluated_image(output_folder)

    def make_predict_bitmaps(self, x, output_folder, cache_dir):
        """
        Desc:   Get the predictions for x and store the respective bitmaps
                in output directory, along with true bitmaps

        input:  cache_dir       :  directory where the cache for the ncfile is
                                    stored
                x               :  input to the keras model
                output_folder   :  output folder to store predictions

        output: y: predicted pixels formatted as Image array.

        """

        y_true = Image.open(os.path.join(
            cache_dir,
            'bitmap_WGS84.bmp')
        ).convert('L')
        y_true_arr = np.asarray(y_true)
        y_pred = self.model.predict(x, batch_size=self.config['batch_size'])

        y_pred_bin = np.where(y_pred > self.predict_thres,
                              1, 0)  # thresholding outputs
        # y_pred_bin  *= y_pred
        # y_pred_bin = y_pred

        y_mat = self.reshape_array_to_image(
            y_pred_bin * 255.0, y_true.size[1], y_true.size[0])

        # store the true and predicted bitmaps in output_dir
        self.save_image(y_mat, os.path.join(output_folder, self.pix_bmp_path))

        # create shapefile output and bitmap from predicted shapefile
        with rasterio.open(os.path.join(cache_dir, BAND_1_FILENAME)) as rasterio_obj:

            convert_bmp_to_shp(
                Image.open(
                    os.path.join(output_folder, self.pix_bmp_path)
                ).convert('L'),
                rasterio_obj.transform,
                os.path.join(output_folder, self.shp_path)
            )
            y_shp = get_bitmap_from_shp(
                os.path.join(output_folder, self.shp_path),
                rasterio_obj,
                os.path.join(output_folder, self.shp_bmp_path),
            )

        self.save_image(y_true_arr, os.path.join(
            output_folder, self.t_bmp_path))

    def reshape_array_to_image(self, dim1_array, x_shape, y_shape):
        """
        desc    : reshape given 1D array to a 2D array of given x,y Dimensions
        """
        return np.asarray(dim1_array, dtype='uint8').reshape((x_shape, y_shape), order='C')

    def save_image(self, img_array, loc):
        """
        Desc    : save given 'img_array' as image in the given 'loc' location
        """
        Image.fromarray(img_array).save(loc)

    def plot_evaluated_image(self, output_folder):
        """
        Desc:   Plot the Evaluation Image with appropriate metrics
                using matplotlib library
                metrics used: Accuracy, precision, recall, IOU score.

        input:  output_folder: folder where the images are to be stored.
        """
        # read true and predict bitmaps
        y_true = mpimg.imread(os.path.join(output_folder, self.t_bmp_path))
        y_shp = mpimg.imread(os.path.join(output_folder, self.shp_bmp_path))
        y_pred = mpimg.imread(os.path.join(output_folder, self.pix_bmp_path))
        b1_img = mpimg.imread(os.path.join(output_folder, self.band_img_path))

        # create BAND_1 AND predicted bitmap subplots
        plt.subplot(2, 1, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(b1_img)
        plt.xlabel('GOES Band 1')
        plt.subplot(2, 1, 2)
        plt.imshow(y_true, alpha=1.0, cmap='plasma')
        plt.imshow(y_shp, alpha=0.5, cmap='PuBu', vmin=125, vmax=255)
        plt.imshow(y_pred, alpha=0.5, cmap='PuBuGn', vmin=125, vmax=255)

        # add confusion matrix to the plot
        a, r, p = self.conf_mtx(y_true, y_pred)
        sa, sr, sp = self.conf_mtx(y_true, y_shp)

        # add IOU score to the plot
        iou = IOU_score(Image.open(os.path.join(output_folder, self.shp_bmp_path), mode='r'),
                        Image.open(os.path.join(output_folder, self.t_bmp_path), mode='r'))

        disp_str = EVAL_DISP_STR.format(a, r, p, iou)
        plt.xlabel(disp_str)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, self.eval_img_path),
                    bbox_inches='tight')

        plt.figure()
        plt.imshow(b1_img)
        plt.imshow(y_true, alpha=0.20, cmap='copper', vmin=0, vmax=255)
        plt.xlabel('True Shapefile Overlaid on BAND 1')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(output_folder, self.true_overlay),
                    bbox_inches='tight')
        plt.close()

    def conf_mtx(self, y_true, y_pred):
        """
        desc:   Give out A,R,P scores for given true,predict
        Input:  y_true:     True Pixels
                y_predict:  Predicted pixels
        Output: acc         :
                recall      :
                precision   :
        """
        cm = confusion_matrix(
            y_true.flatten() > self.predict_thres * 255.0,
            y_pred.flatten() > self.predict_thres * 255.0
        )

        overall_true = float(cm[0][0] + cm[1][1])
        total = float(cm[0][0] + cm[0][1] + cm[1][1] + cm[1][0])
        true_positive = float(cm[1][1])
        actual_positive = float(cm[1][0] + cm[1][1])
        predicted_positive = float(cm[0][1] + cm[1][1])

        acc = self.safe_div(overall_true, total)
        recall = self.safe_div(true_positive, actual_positive)
        precision = self.safe_div(true_positive, predicted_positive)

        return acc, recall, precision

    def safe_div(self, x, y):
        if y == 0:
            return 0

        return x / y


if __name__ == '__main__':

    ev = Evaluate(json.load(open('config.json')))
    ev.evaluate()
