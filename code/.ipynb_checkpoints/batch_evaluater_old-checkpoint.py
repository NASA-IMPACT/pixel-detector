import matplotlib
matplotlib.use('Agg')


from glob import glob
from data_preparer import PixelDataPreparer, PixelListPreparer
from tensorflow.keras.models import load_model
from PIL import Image
from shape_utils import (
    bitmap_from_shp,
)
from config import (
    PREDICT_THRESHOLD,
)

import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import rasterio
import numpy.ma as ma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# data = np.random.random((10, 10))
# data = 10 * (data - 0.8)

# fig, ax = plt.subplots()
# norm = MidpointNormalize(midpoint=0)
# im = ax.imshow(data, norm=norm, cmap=plt.cm.seismic, interpolation='none')
# fig.colorbar(im)
# plt.show()


class Evaluate:

    def __init__(self, num_n):
        """init for Evaluate class
        """
        # self.model = load_model("../models/smoke_unet.h5")
        self.model = load_model('../models/smoke_pixel.h5')
        self.conf_mtx = np.zeros((2, 2))
        self.save_dir = "../data/test_unet"
        path = "../unet_master/val/frames/data/"
        path_list = glob(path + '/*.tif')
        step_num = 5
        #paths_lists = [path_list[x:x + step_num] for x in range(0, len(path_list), step_num)]
        #for paths in paths_lists:

        self.dataset = PixelListPreparer(path_list, neighbour_pixels=num_n)
        self.dataset.iterate([0,1,2,3,4,5])

        self.evaluate()

    def evaluate(self):
        """
        evaluate workflow
        """
        input_data = self.dataset.dataset
        labels = self.dataset.labels
        predicted_pixels = self.get_predictions(input_data, labels)
        #self.plot_predictions(self.dataset, predicted_pixels)

        # def get_predictions(self, input_data, labels):
        #     """return predictions for given input data
        #     Args:
        #         input_data (TYPE): input data
        #     Returns:
        #         TYPE: prediction array list
        #     """
        #     last_idx = 0
        #     prediction_list = []
        #     conf_mtx = np.zeros((2, 2))
        #     total_images = len(self.dataset.img_dims_list)
        #     for i, img_shape in enumerate(self.dataset.img_dims_list):
        #         print('predicting {} of {} images:'.format(
        #             i + 1, total_images))
        #         img_prediction = self.__predict__(
        #             input_data[last_idx:last_idx + img_shape[0] * img_shape[1]])
        #         img_prediction = self.__predict__(
        #             input_data[last_idx:last_idx + img_shape[0] * img_shape[1]])
        #         img_true = labels[last_idx:last_idx + img_shape[0] * img_shape[1]]
        #         conf_mtx += confusion_matrix(img_true,
        #                                      img_prediction > PREDICT_THRESHOLD,)
        #         last_idx += img_shape[0] * img_shape[1]
        #         prediction_list.append(img_prediction.reshape(
        #             (img_shape[0], img_shape[1])))

        #     # save confusion matrix
        #     df_cm = pd.DataFrame(conf_mtx, range(2),
        #                          range(2))
        #     sn.set(font_scale=1.4)  # for label size
        #     cf = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        #     cf.get_figure().savefig('conf_mat_pixel.png')

        #     # save confusion matrix
        #     tp = conf_mtx[1][1]
        #     tn = conf_mtx[0][0]
        #     fp = conf_mtx[0][1]
        #     fn = conf_mtx[1][0]

        #     sensitivity = tp / (tp + fn)
        #     specificity = tn / (tn + fp)
        #     precision = tp / (tp + fp)
        #     recall = tp / (tp + fn)
        #     accuracy = (tp + tn) / (tp + tn + fp + fn)
        #     f1_score = 2 * ((precision * recall) / (precision + recall))

        #     metrics = dict()
        #     metrics['tp'] = tp
        #     metrics['tn'] = tn
        #     metrics['fp'] = fp
        #     metrics['fn'] = fn
        #     metrics['sensitivity'] = sensitivity
        #     metrics['specificity'] = specificity
        #     metrics['precision'] = precision
        #     metrics['recall'] = recall
        #     metrics['accuracy'] = accuracy
        #     metrics['f1_score'] = f1_score

        #     with open('{}_metrics.txt'.format('pixel'), 'w') as txt:
        #         txt.write(str(metrics))
        #     return prediction_list


    def get_predictions(self, input_data, labels):
        """return predictions for given input data
        Args:
            input_data (TYPE): input data
        Returns:
            TYPE: prediction array list
        """
        last_idx = 0
        prediction_list = []
        total_images = len(self.dataset.img_dims_list)
        for i, img_shape in enumerate(self.dataset.img_dims_list):
            print('predicting {} of {} images:'.format(
                i + 1, total_images))
            img_prediction = self.__predict__(
                input_data[last_idx:last_idx + img_shape[0] * img_shape[1]])
            img_true = labels[last_idx:last_idx + img_shape[0] * img_shape[1]]
            self.conf_mtx += confusion_matrix(img_true,
                                         img_prediction > PREDICT_THRESHOLD,)
            last_idx += img_shape[0] * img_shape[1]
            prediction_list.append(img_prediction.reshape(
                (img_shape[0], img_shape[1])))

        # # save confusion matrix
        # df_cm = pd.DataFrame(self.conf_mtx, range(2),
        #                      range(2))
        # sn.set(font_scale=1.4)  # for label size
        # cf = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        # cf.get_figure().savefig('conf_mat_pixel.png')

        # save confusion matrix
        tp = self.conf_mtx[1][1]
        tn = self.conf_mtx[0][0]
        fp = self.conf_mtx[0][1]
        fn = self.conf_mtx[1][0]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        metrics = dict()
        metrics['tp'] = tp
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['sensitivity'] = sensitivity
        metrics['specificity'] = specificity
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['accuracy'] = accuracy
        metrics['f1_score'] = f1_score

        with open('{}_metrics.txt'.format('pixel'), 'w') as txt:
            txt.write(str(metrics))
        return prediction_list


    def __predict__(self, data):
        """give out predictions for the given image
        Returns:
            TYPE: Description
        """
        pred_bmp = self.model.predict(np.array(data), batch_size=50000, verbose=1)
        return pred_bmp

    def plot_predictions(self, dataset, predictions, ):
        """plot  prediction heatmaps along with original image
        for evaluation purposes
        Args:
            x_path (TYPE): list of image paths
            predictions (TYPE): numpy arrays of predictions
        """
        x_path = dataset.img_path_list
        transforms = dataset.raster_transforms

        assert len(x_path) == len(predictions)
        for i, prediction in enumerate(predictions):
            img_name = os.path.basename(x_path[i])
            print('plotting {} of {} images:'.format(i + 1, len(predictions)))
            #fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

            with rasterio.open(x_path[i]) as rast:
                red = rast.read(2)
                blue = rast.read(1)
                pseudo_green = rast.read(3)
                height, width = red.shape
                true_green = np.array((0.45 * red.astype('float32')) +
                                      (0.1 * pseudo_green.astype('float32')) +
                                      (0.45 * blue.astype('float32')),
                                      dtype='uint8',
                                      )

                img = np.moveaxis(
                    np.array([red, pseudo_green, blue]), 0, -1
                )
            fig = plt.figure()
            fig.set_size_inches(width / height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(self.save_dir,
                                     img_name.replace('.tif', '.png')), dpi=height)
            thres_binary = np.asarray((prediction > PREDICT_THRESHOLD) * 255,
                                      dtype='uint8')

            thres_pred = np.asarray(prediction * 255,
                                    dtype='uint8')

            thres_pred_masked = ma.masked_where(thres_pred <= PREDICT_THRESHOLD * 255.0,
                                                thres_pred)

            plt.imshow(thres_pred_masked, alpha=0.35, cmap='spring')
            plt.axis('off')
            # Image.fromarray(thres_pred.astype(np.uint8)).save(
            #     '{}_bmp.bmp'.format(i))
            vis_path = os.path.join(self.save_dir, img_name.replace('.tif', 'predict.bmp'))
            # bitmap_from_shp( thres_pred,
            #                transforms[i],
            #                os.path.join(self.save_dir, str(i)),
            #                vis_path,
            #                )
            plt.savefig(os.path.join(self.save_dir,
                                     img_name.replace('.tif', '_masked.png')), dpi=height)
            Image.fromarray(thres_binary).convert('L').save(vis_path)

def IoU(y_true, y_pred, eps=1e-6):

        if np.max(y_true) == 0.0:
            return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
        return -K.mean( (intersection + eps) / (union + eps), axis=0)


if __name__ == '__main__':
    import json
    config = json.load(open('config.json'))
    ev = Evaluate(num_n=7)

