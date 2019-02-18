# @author Muthukumaran R.

import json
import numpy as np
import os
import rasterio

from keras.models import load_model
from PIL import Image


from config import (
    PREDICT_THRESHOLD,
    OUTPUT_DIR,
)
from preprocessing import (
    get_arrays_from_json,
    get_arrays_for_prediction,
    convert_bmp_to_shp,
)


class Predict:

    def __init__(self, config):
        """
            config = {
                'type'          : < choice of 'pixel' or 'deconv'>,
                'num_neighbor'  : < n of the n*n matrix to take as
                                    input to the model>

                'model_path'    : path to the keras model.

                'jsonfile'      : <json containing ncfile, extent
                                    to subset ncfile and shapefile
                                    for training >

                'num_epoch'     : keras num_epoch

                'batch_size'    : keras batch_size

                'eval_json'     : <json containing ncfile, extent
                                    to subset ncfile and shapefile
                                    for evaluation >

                'pred_json'     : <json containing ncfile and extent
                                    to subset for prediction>
            }
        """

        self.config = config
        self.jsonfile = config['pred_json']
        self.model = load_model(str(self.config['model_path']))
        self.threshold = PREDICT_THRESHOLD
        self.shp_path = config['pred_shp_path']

    # def predict(self, shp_path):

        """
        Desc        : returns shapefile object
        input       : None
        Output      : fiona shapefile objects List TODO: return Geojson dict

        """

    def predict(self):
        """
        Desc        : writes the predicted smoke shapefile to shp_path
        input       : Path to store shapefiles

        """

        x_list, transform_list = get_arrays_for_prediction(self.config['pred_json'],
                                                           self.config['num_neighbor'])

        for id_, (x, transform_tuple) in enumerate(unzip(x_list, transform_list)):

            raster_transform, res = transform_tuple

            # predict for x
            y_pred = self.model.predict(
                x, batch_size=self.config['batch_size'])

            y_pred = y_pred > self.threshold

            # TODO: checks for reshape needed.
            y_mat = np.asarray(y_pred * 255, dtype='uint8').reshape((res[1],
                                                                     res[0]), order='C')

            print('generating shapefiles...')
            convert_bmp_to_shp(Image.fromarray(y_mat).convert('L'),
                               raster_transform,
                               self.shp_path + str(id_)
                               )


if __name__ == '__main__':

    pred = Predict(json.load(open('config.json')))
    # pred.predict('/nas/rhome/mramasub/smoke_pixel_detector/data/prod_level/')
    pred.predict()
