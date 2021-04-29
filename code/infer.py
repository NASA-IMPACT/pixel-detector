from tensorflow.keras.models import Model, load_model
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
import cv2
import xarray
from pyorbital import astronomy

matplotlib.use('Agg')

from math import floor, ceil
from pyproj import Proj
from rasterio.windows import from_bounds


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


def longlat2window(lon, lat, dataset):
    """
    Args:
        lon (tuple): Tuple of min and max lon
        lat (tuple): Tuple of min and max lat
        dataset: Rasterio dataset

    Returns:
        rasterio.windows.Window
    """
    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t * (xmin, ymin)
    col_max, row_max = ~t * (xmax, ymax)
    return Window.from_slices(rows=(floor(row_max), ceil(row_min)),
                              cols=(floor(col_min), ceil(col_max)))


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


def convert_rgb(img):

    red = img[:, :, 1]
    blue = img[:, :, 0]
    pseudo_green = img[:, :, 2]
    height, width = red.shape
    img = np.moveaxis(
        np.array([red, pseudo_green, blue]), 0, -1
    )

    return img



def check_raw_file():
    pass

def rad_to_ref(memfile, utc_time, cza_correct=True, gamma=2.0):

    ref_list = []
    data_array = xarray.open_rasterio(memfile)
    for i, data in enumerate(data_array[:-1]):
        rad = data.data * KAPPA0[i]
        if cza_correct:
            x, y = np.meshgrid(data_array['x'], data_array['y'])
            cza = astronomy.cos_zen(utc_time, x, y)
            rad = rad * cza
        ref = np.clip(rad, 0, 1)
        ref_clipped = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)
        ref_list.append(ref_clipped)
    data_array.close()
    return np.array(ref_list, dtype='uint8')


def run_comparision():
    model = load_model('../models/smoke_unet_iou_aug.h5',
                       custom_objects={'IoU': IoU})

    dev_source = 'test_2020/time-20202511900196-loc--125.0_30.0_-110.0_40.0.tif'
    IMG_RES = 256
    date_time = '202025119'
    extent = CONUS_EXTENT_COORDS
    scene_id = SCENE_ID_FORMAT.format(date_time)
    start_x, start_y, end_x, end_y = calculate_tile_xy(extent)
    extent = calculate_new_bbox(start_x, start_y, end_x, end_y)
    height = (end_x - start_x + 1) * TILE_SIZE
    width = (end_y - start_y + 1) * TILE_SIZE
    transform = rasterio.transform.from_bounds(
        *extent, height, width
    )
    final_predictions = np.zeros((width, height))
    with rasterio.open(dev_source) as src:
        for y in range(start_y, end_y + 1):
            start_index_y = (y - start_y) * TILE_SIZE
            for x in range(start_x, end_x + 1):
                start_index_x = (x - start_x) * TILE_SIZE
                tile_url = WMTS_URL.format(ZOOM_LEVEL, x, y, scene_id)
                response = requests.get(tile_url)
                with open(f'test_tifs/{y}_{x}.tif', "wb") as f:
                    f.write(response.content)
                memoryfile = rasterio.io.MemoryFile(response.content).open()
                wmts_array = memoryfile.read()[:-1]
                src_array = src.read(
                    window=from_bounds(
                        *memoryfile.bounds,
                        src.transform,
                        width=256,
                        height=256,
                    )
                )
                wmts_array = rad_to_ref(memoryfile, '2020-09-07T19:05:05.086315008')
                vis_band = 1
                f, ax = plt.subplots(1, 2, constrained_layout=True, dpi=100)
                ax[0].imshow(convert_rgb(cv2.resize(np.moveaxis(wmts_array, 0, -1), (256,256))))
                ax[0].set_title('WMTS')
                ax[0].xaxis.set_ticks([])
                ax[0].yaxis.set_ticks([])
                # ax[1].imshow(src_array[vis_band].astype('uint8'))
                ax[1].imshow(convert_rgb(cv2.resize(np.moveaxis(src_array, 0, -1), (256,256))))
                ax[1].xaxis.set_ticks([])
                ax[1].yaxis.set_ticks([])
                ax[1].set_title('ncfile')
                predictions = predict(
                    wmts_array, model
                )
                ax[0].imshow(ma.masked_where(predictions < 0.5, predictions),alpha=0.20,cmap='bwr')
                predictions = predict(
                    src_array, model
                )
                ax[1].imshow(ma.masked_where(predictions < 0.5, predictions),alpha=0.20,cmap='bwr')
                plt.savefig(f'testfiles/{y}_{x}.png')
                plt.close()

def convert_rgb(img):
    red = img[:, :, 1 ]
    blue = img[:, :, 0 ]
    pseudo_green = img[:, :, 2]
    img = np.moveaxis(
        np.array([red, pseudo_green, blue]), 0, -1
    )

    return img.astype('uint8')

if __name__ == '__main__':
    run_comparision()
