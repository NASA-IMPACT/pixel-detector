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
        self.config         = config
        self.predict_thres  = PREDICT_THRESHOLD
        self.output_path    = OUTPUT_DIR
        self.eval_img_path  = 'evaluation_plot.png'
        self.band_img_path  = 'band1.png'
        self.pix_bmp_path   = 'pred_pix_bitmap.bmp'
        self.shp_bmp_path   = 'pred_shp_bitmap.bmp'
        self.t_bmp_path     = 'true_bitmap.bmp'
        self.shp_path       = 'shapefile.shp'
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
        id_ = 0
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

            self.plot_evaluated_image(output_folder, id_)




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

            y_pred_bin  = np.where(y_pred > self.predict_thres, 1, 0) #thresholding outputs
            #y_pred_bin  *= y_pred
            #y_pred_bin = y_pred

            y_mat = np.asarray(y_pred_bin*255.0,dtype = 'uint8').reshape((y_true.size[1],
                y_true.size[0]),order='C')

            # store the true and predicted bitmaps in output_dir
            Image.fromarray(y_mat).save(os.path.join(output_folder, self.pix_bmp_path))

            # create shapefile output and bitmap from predicted shapefile
            with rasterio.open(os.path.join(cache_dir,'0_WGS84.tif')) as rasterio_obj:

                convert_bmp_to_shp(Image.open(os.path.join(output_folder, self.pix_bmp_path)).convert('L'),
                                    rasterio_obj.transform,
                                    os.path.join(output_folder, self.shp_path)
                                    )

                y_shp = get_bitmap_from_shp(os.path.join(output_folder, self.shp_path),
                                            rasterio_obj,
                                            os.path.join(output_folder, self.shp_bmp_path)
                                            )

            Image.fromarray(y_true_arr).save(os.path.join(output_folder, self.t_bmp_path))





    def plot_evaluated_image(self, output_folder, id_):

        """
        Desc:   Plot the Evaluation Image with appropriate metrics
                using matplotlib library
                metrics used: Accuracy, precision, recall, IOU score.

        input:  output_folder: folder where the images are to be stored.
        """
        # read true and predict bitmaps
        y_true = mpimg.imread(os.path.join(output_folder, self.t_bmp_path))
        y_shp  = mpimg.imread(os.path.join(output_folder, self.shp_bmp_path))
        y_pred = mpimg.imread(os.path.join(output_folder, self.pix_bmp_path))
        b1_img = mpimg.imread(os.path.join(output_folder, self.band_img_path))

        # create BAND_1 AND predicted bitmap subplots
        plt.subplot(2,1,1)
        plt.imshow(b1_img)
        plt.xlabel('GOES Band 1')
        plt.subplot(2,1,2)
        plt.imshow(y_true, alpha=1.00, cmap='plasma')
        plt.imshow(y_shp,  alpha=0.5, cmap='PuBu', vmin = 125,vmax=255)
        plt.imshow(y_pred, alpha=0.5, cmap='PuBuGn', vmin = 125,vmax=255)


        #add confusion matrix to the plot
        a,r,p    = self.conf_mtx(y_true,y_pred)
        sa,sr,sp = self.conf_mtx(y_true,y_shp)

        #add IOU score to the plot
        iou = IOU_score(Image.open(os.path.join(output_folder, self.shp_bmp_path), mode='r'),
                        Image.open(os.path.join(output_folder, self.t_bmp_path), mode='r'))

        disp_str = 'A:{0:.2f},R:{1:.2f},P:{2:.2f},IOU:{3:.2f}\nSA:{0:.2f},SR:{1:.2f},SP:{2:.2f}:'\
        .format(a,r,p,iou,sa,sr,sp)
        plt.xlabel(disp_str)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,self.eval_img_path))
        plt.close()



    def conf_mtx(self,y_true,y_pred):
        """
        desc:   Give out A,R,P scores for given true,predict
        Input:  y_true:     True Pixels
                y_predict:  Predicted pixels
        Output: acc         :
                recall      :
                precision   :
        """
        cm = confusion_matrix(
            y_true.flatten() > self.predict_thres*255.0,
            y_pred.flatten() > self.predict_thres*255.0
            )

        acc         = self.safe_div(float(cm[0][0] + cm[1][1]), float(cm[0][0]+cm[0][1]+cm[1][1] +cm[1][0]))
        recall      = self.safe_div(float(cm[1][1]), float(cm[1][0] +cm[1][1]))
        precision   = self.safe_div(float(cm[1][1]), float(cm[0][1] +cm[1][1]))

        return acc, recall, precision

    def safe_div(self,x,y):
        if y == 0:
            return 0

        return x / y

if __name__ == '__main__':


    eval = Evaluate(json.load(open('config.json')))
    eval.evaluate()



