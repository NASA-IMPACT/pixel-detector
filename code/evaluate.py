# @author Muthukumaran R.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json
import matplotlib
import numpy as np

from keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
import rasterio
import os

from config import (
    PREDICT_THRESHOLD,
    OUTPUT_DIR
    )
from preprocessing import (
            get_arrays_from_json,
            convert_bmp_to_shp,
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

                'eval_json'     : <json containing ncfile, extent
                                    to subset ncfile and shapefile
                                    for evaluation >

                'pred_json'     : <json containing ncfile and extent
                                    to subset for prediction>
            }
        """
        self.config = config
        self.predict_thres = PREDICT_THRESHOLD
        self.output_path = OUTPUT_DIR
        self.eval_img_path = 'eval.png'
        self.band_img_path = 'band1.png'
        self.p_bmp_path = 'predict_bitmap.bmp'
        self.t_bmp_path = 'true_bitmap.bmp'
        self.shp_path = 'eval.shp'
        #self.model_holder = MODELS[self.config['type']] (self.config)



    def evaluate(self):
        """
        desc:   produce images containing predictions along with their scores
                scores produced: Accuracy, precision, recall, IOU score.
        """

        print('looking for model in', str(self.config['model_path']))
        self.model = load_model(str(self.config['model_path']))

        x_list,y_list,cache_list = get_arrays_from_json(self.config['eval_json'],
                                         self.config['num_neighbor']
                                        )
        for x,y,cache_dir in zip(x_list,y_list,cache_list):
            # iterate through each x,y pair

            output_folder = os.path.join(self.output_path,
                os.path.basename(os.path.normpath(cache_dir)))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            self.make_predict_bitmaps(x,output_folder, cache_dir)

            # save BAND 1 raster from cache for evaluation purposes
            b1_rast = Image.open(os.path.join(cache_dir,'0_WGS84.tif')).convert('RGBA')
            b1_rast.save(os.path.join(output_folder, self.band_img_path))

            self.plot_evaluated_image(output_folder)

            # create shapefile output
            with rasterio.open(os.path.join(cache_dir,'0_WGS84.tif')) as rasterio_obj:
                convert_bmp_to_shp(Image.open(os.path.join(output_folder, self.p_bmp_path)).convert('L'),
                                    rasterio_obj.transform,
                                    os.path.join(output_folder, self.shp_path)
                                    )




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

            y_true      = Image.open(os.path.join(cache_dir,'bitmap_WGS84.bmp')).convert('L')
            y_true_arr  = np.asarray(y_true)
            y_pred      = self.model.predict(x,batch_size = self.config['batch_size'])
            y_pred_bin  = y_pred > self.predict_thres #thresholding outputs
            #y_pred_bin = y_pred

            y_mat = np.asarray(y_pred_bin*255.0,dtype = 'uint8').reshape((y_true.size[1],
                y_true.size[0]),order='C')

            # store the true and predicted bitmaps in output_dir
            Image.fromarray(y_mat).save(os.path.join(output_folder, self.p_bmp_path))
            Image.fromarray(y_true_arr).save(os.path.join(output_folder, self.t_bmp_path))





    def plot_evaluated_image(self, output_folder):

        """
        Desc:   Plot the Evaluation Image with appropriate metrics
                using matplotlib library
                metrics used: Accuracy, precision, recall, IOU score.

        input:  output_folder: folder where the images are to be stored.
        """
        # read true and predict bitmaps
        y_true = mpimg.imread(os.path.join(output_folder, self.t_bmp_path))
        y_pred = mpimg.imread(os.path.join(output_folder, self.p_bmp_path))
        b1_img = mpimg.imread(os.path.join(output_folder, self.band_img_path))

        # create BAND_1 AND predicted bitmap subplots
        plt.subplot(1,2,1)
        plt.imshow(b1_img)
        plt.xlabel('GOES Band 1')
        plt.subplot(1,2,2)
        plt.imshow(y_pred, cmap='jet')
        plt.imshow(y_true, alpha=0.45, cmap='gray')

        # add confusion matrix to the plot
        cm = confusion_matrix(
            y_true.flatten() > self.predict_thres*255.0,
            y_pred.flatten() > self.predict_thres*255.0
            )
        acc         = float(cm[0][0] + cm[1][1]) / float(cm[0][0]+cm[0][1]+cm[1][1] + cm[1][0])
        recall      = float(cm[1][1]) / float(cm[1][0] +cm [1][1])
        precision   = float(cm[1][1]) / float(cm[0][1]+cm[1][1])

        #add IOU score to the plot
        iou = IOU_score(Image.open(os.path.join(output_folder, self.p_bmp_path), mode='r'),
                        Image.open(os.path.join(output_folder, self.t_bmp_path), mode='r'))

        disp_str = 'a:{0:.2f},r:{1:.2f},p:{2:.2f},IOU:{3:.2f}'.format(acc,recall,precision,iou)
        plt.xlabel('True Smoke over Predicted Smoke\n'+disp_str)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,self.eval_img_path))
        plt.close()





if __name__ == '__main__':


    eval = Evaluate(json.load(open('config.json')))
    eval.evaluate()



