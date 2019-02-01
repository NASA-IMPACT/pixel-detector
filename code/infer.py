
# @author Muthukumaran R.

from preprocessing import get_arrays_for_prediction,convert_bmp_to_shp
from keras.models import load_model
from PIL import Image

import datetime
import numpy as np

class Predicter:

    def __init__(self,ncfile,time,extent,model_path):

        nctime = self.get_timestr(time)
        print("time now",nctime)
        nctime = "20180822000"
        self.jsondict = [{"ncfile":ncfile,
                         "nctime":nctime,
                         "extent":extent
                        }]
        self.threshold      = 0.5
        self.num_neighbor   = 5
        self.model          = load_model(model_path)
        self.x, transform_holder = get_arrays_for_prediction(self.jsondict,
            self.num_neighbor)
        self.rast_transform,self.res    = transform_holder[0]


    def predict(self):
        """
        """

        y_pred = self.model.predict(self.x[0],batch_size = 10000)

        y_pred = y_pred > self.threshold

        # TODO: checks for reshape needed.
        y_mat = np.asarray(y_pred*255,dtype = "uint8").reshape((self.res[1],
            self.res[0]),order="C")

        geojson_dict = convert_bmp_to_shp(  Image.fromarray(y_mat).convert("L"),
                                            self.rast_transform,"")

        return geojson_dict




    def get_timestr(self,datetime_obj):

        print(datetime_obj.timetuple().tm_yday)
        day_of_year = str(datetime_obj.timetuple().tm_yday)

        return str(datetime_obj.year)+day_of_year+str(datetime_obj)




if __name__ == "__main__":

    ncfile = "/nas/rgroup/dsig/smoke/goes_data_qc/082/20"
    time   = datetime.datetime.now()
    extent =    [   -84.7,
                    22.6,
                    -81.0,
                    26.4
                 ]
    model  = "../models/fully_dense_model_experiment.h5"

    pred_instance   = Predicter(ncfile,time,extent,model)
    geojson_dict    = pred_instance.predict()

    print(geojson_dict)