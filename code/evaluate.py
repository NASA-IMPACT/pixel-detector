# @author Muthukumaran R.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json
import matplotlib
import numpy as np
from config import (
    PREDICT_THRESHOLD,
    OUTPUT_DIR
    )
from keras.models import load_model
from PIL import Image
from preprocessing import (
            get_arrays_from_json,
            convert_bmp_to_shp,
            IOU_score
            )
from sklearn.metrics import confusion_matrix
import rasterio
import os


class Evaluate:

    def __init__(self, config):
        """
        config = {
            'type': < choice of 'pixel' or 'deconv'>,
            'num_neighbor': < TODO: explain this parameter >,
            ''
        }
    """

        self.config = config
        self.predict_thres = PREDICT_THRESHOLD
        self.output_folder = OUTPUT_DIR
        self.eval_img_path = 'eval.png'
        self.band_img_path = 'band1.png'
        self.p_bmp_path = 'predict_bitmap.bmp'
        self.t_bmp_path = 'true_bitmap.bmp'
        self.shp_path = 'eval.shp'
        #self.model_holder = MODELS[self.config['type']] (self.config)

    def evaluate(self):
        """
        desc:
        """
        print(str(self.config['savepath']))
        model = load_model(str(self.config['savepath']))
        x_list,y_list,cache_dir_list = get_arrays_from_json(self.config['eval_json'],5, shuffle=False)

        for x,y,cache_dir in zip(x_list,y_list,cache_dir_list):

            # iterate through each result and store them in
            output_path = os.path.join(self.output_folder,
                os.path.basename(os.path.normpath(cache_dir)))

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            y_true = Image.open(os.path.join(cache_dir,'bitmap_WGS84.bmp')).convert('L')
            y_true_arr = np.asarray(y_true)
            b1_rast = Image.open(os.path.join(cache_dir,'0_WGS84.tif')).convert('RGBA')
            y_pred = model.predict(x,batch_size = 10000)
            y_pred_bin = y_pred > self.predict_thres
            #y_pred_bin = y_pred
            y_mat = np.asarray(y_pred_bin*255.0,dtype = 'uint8').reshape((y_true.size[1],
                y_true.size[0]),order='C')


            Image.fromarray(y_mat).save(os.path.join(output_path, self.p_bmp_path))
            Image.fromarray(y_true_arr).save(os.path.join(output_path, self.t_bmp_path))

            b1_rast.save(os.path.join(output_path, self.band_img_path))

            self.plot_img(output_path,y)

            with rasterio.open(os.path.join(cache_dir,'0_WGS84.tif')) as rasterio_obj:
                convert_bmp_to_shp(os.path.join(output_path, self.p_bmp_path),
                                    rasterio_obj.transform,
                                    os.path.join(output_path, self.shp_path)
                                    )




    def plot_img(self, output_folder,y):
        """
        put confusion matrix text on image

        """

        y_true = mpimg.imread(os.path.join(output_folder, self.t_bmp_path))
        y_pred = mpimg.imread(os.path.join(output_folder, self.p_bmp_path))
        b1_img = mpimg.imread(os.path.join(output_folder, self.band_img_path))

        plt.subplot(1,2,1)
        plt.imshow(b1_img)
        #plt.imshow(y_true, alpha=0.05, cmap='gray')
        plt.xlabel('GOES Band 1')

        #temp_y_out = y_pred > PREDICT_THRESHOLD
        temp_y_out = y_pred
        plt.subplot(1,2,2)
        plt.imshow(y_pred, cmap='jet')
        plt.imshow(y_true, alpha=0.45, cmap='gray')

        # add confusion matrix
        cm = confusion_matrix(y,y_pred.flatten() > self.predict_thres*255.0)
        acc = float(cm[0][0] + cm[1][1]) / float(cm[0][0]+cm[0][1]+cm[1][1] + cm[1][0])
        recall = float(cm[1][1]) / float(cm[1][0] +cm [1][1])
        precision = float(cm[1][1]) / float(cm[0][1]+cm[1][1])

        #add IOU score
        iou = IOU_score(os.path.join(output_folder, self.p_bmp_path),
            os.path.join(output_folder, self.t_bmp_path))
        disp_str = 'a:{0:.2f},r:{1:.2f},p:{2:.2f},IOU:{3:.2f}'.format(acc,recall,precision,iou)
        plt.xlabel('True Smoke over Predicted Smoke\n'+disp_str)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,self.eval_img_path))
        plt.close()





if __name__ == '__main__':
    # build model

    eval = Evaluate(json.load(open('config.json')))
    eval.evaluate()
