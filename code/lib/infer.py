import cv2
import datetime
import matplotlib
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import numpy.ma as ma
import rasterio
import requests
import tensorflow.keras.backend as K
import urllib
import xarray

from math import floor, ceil
from pyorbital import astronomy
from pyproj import Proj

from data_rasterizer_wmts import calculate_tile_xy

from tensorflow.keras.models import Model, load_model

from rasterio.windows import from_bounds

matplotlib.use('Agg')


KAPPA0 = [
    0.0015810000477358699,
    0.001954900100827217,
    0.003332100110128522,
    0.008836500346660614,
    0.013148699887096882,
    0.04147079959511757,
]

# Model Constants
CONUS_EXTENT_COORDS = [-125.0, 30.0, -110.0, 40.0]
FULLDISK_EXTENT_COORDS = [-149.765109632, -
                          64.407370418, -0.234889169, 64.407370030]
GAMMA = 2.0
IMAGE_BUCKET = 'phenom-detect-internal'
INTERSECTION_THRES = 1.0

MIN_POINTS = 3
NUM_NEIGHBOR = 7
PREDICT_THRESHOLD = 0.5
SCALE_FACTOR = 1.0
SCENE_ID_FORMAT = 'OR_ABI-L1b-RadF-M6_G16_s{}'
THRESHOLD = 0.5

TRUE_COLOR_EXPRESSION = urllib.parse.quote("B02,0.45*B02+0.1*B03+0.45*B01,B01")
TILE_SIZE = 256
WMTS_BASE_URL = "https://ffasjnxf3l.execute-api.us-east-1.amazonaws.com/production/"
# WMTS_EXPRESSION_URL = f"{WMTS_BASE_URL}epsg4326/expression/{{}}/{{}}/{{}}.png?sceneid={{}}&expr={TRUE_COLOR_EXPRESSION}&rescale=0,255&color_ops={COLOR_OPS}"
WMTS_URL = f"{WMTS_BASE_URL}epsg4326/{{}}/{{}}/{{}}.tif?sceneid={{}}"
ZOOM_LEVEL = 7

META_URL = "{}{}".format(WMTS_BASE_URL, "metadata?sceneid={}")


def calculate_new_bbox(start_x, start_y, end_x, end_y):
    start_lon, _, _, end_lat = mercantile.bounds(
        start_x, start_y, ZOOM_LEVEL
    )
    _, start_lat, end_lon, _ = mercantile.bounds(
        end_x, end_y, ZOOM_LEVEL
    )
    print(start_lon, start_lat, end_lon, end_lat)
    return [start_lon, start_lat, end_lon, end_lat]


def calculate_tile_xy(extent):
    start_x, start_y, _ = mercantile.tile(
        extent[0],
        extent[3],
        ZOOM_LEVEL
    )

    end_x, end_y, _ = mercantile.tile(extent[2], extent[1], ZOOM_LEVEL)
    return [start_x, start_y, end_x, end_y]


def IoU(y_true, y_pred, eps=1e-6):

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + \
        K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)


def predict(array, model):
    array = cv2.resize(
            np.moveaxis(array, 0, -1), (256,256))
    y_pred = model.predict(np.array([array]))

    return y_pred[0,:,:,0]


def combine_rasters(img_list, b1_meta, save_path):

    b1_meta.update(
        {
            'count': len(img_list),
            'driver': 'GTiff',
            'crs': {'init': 'epsg:4326'},
            'nodata': 0,
            'dtype': 'uint8',
        }
    )
    with rasterio.open(s3_list[0], 'r') as src:
        dest_meta = src.meta
    with rasterio.open(save_path, 'w', **b1_meta) as dest:
        for band_num, band in enumerate(img_list):
            dest.write(band, band_num + 1)


if __name__ == '__main__':
    run_comparision()
