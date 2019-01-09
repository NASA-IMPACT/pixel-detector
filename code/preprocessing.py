# @author Muthukumaran R.

import fiona
import json
import math
import numpy as np
import os
import rasterio
import rasterio.features

# from config import shapefile_path,raw_data_path
from config import (
    BANDS_LIST,
    FULLDISK_EXTENT_COORDS,
    FULL_RES_FD
)
from PIL import Image
from tif_utils import create_array_from_nc, band_list


def convert_pixels_to_groups(img, edge_size=5, stride=0, num_bands=8):
    """
    Given img[x,y] array, the method yields x*y arrays of edge_size*edge_size
    matrices, each array in output corresponds to each pixel in input
    """
    rows, cols, bands = img.shape
    result = []
    half_edge = int(edge_size / 2)
    # if stride > 0:
    #     raise notImplementedError

    for i in range(rows):
        for j in range(cols):
            moving_window = np.zeros((edge_size, edge_size, num_bands), dtype=float)
            moving_window[half_edge + 1, half_edge + 1] = img[i, j]

            # TODO: row_extent and height_extent unused. remove?
            row_extent = i if i - half_edge < 0 else 0
            height_extent = j if j - half_edge < 0 else 0

            if i - half_edge < 0 or j - half_edge < 0 or i + half_edge + 1 > rows or j + half_edge + 1 > cols:
                moving_window[half_edge + 1, half_edge + 1] = img[i, j]
                result.append(np.array(moving_window, dtype=float))

            else:
                result.append(np.array(img[i - half_edge:i + half_edge + 1,
                                           j - half_edge:j + half_edge + 1], dtype=float))

    return np.array(result)


def get_arrays_from_json(jsonfile, num_neighbor, shuffle=True):
    """
    num_neighbor = number of pixels at the egde of the square matrix input
    jsonfile = {
      "ncfile": "<ppath/to/ncfile>",
      "nctime": "unique time string of the ncfile",
      "shp": "<path/to/shapefile>",
      "extent": [extent coordinates in lat/lon],
      "start": "start time string of the shapefile",
      "end": "end time string of the shapefile"
    }

    """

    print('reading json from :', jsonfile)

    js = open(jsonfile)
    jsondict = json.loads(js.read())

    x_array = []
    y_array = []

    for item in jsondict:

        ncpath = item['ncfile']
        nctime = item['nctime']
        nclist = band_list(ncpath, BANDS_LIST, time=nctime)
        extent = item['extent']
        shapefile_path = item['shp']

        # TODO: start and end unused.
        start = item['start']
        end = item['end']

        res = get_res_for_extent(extent, num_neighbor)
        ext_str = '_{}_{}_{}_{}'.format(extent[0], extent[1], extent[2], extent[3])
        arr, tifpath = create_array_from_nc(nclist, fname=nctime + ext_str, extent=extent, res=res)
        grp_array = convert_pixels_to_groups(arr)

        if x_array == []:
            x_array = grp_array

        else:
            x_array = np.append(x_array, grp_array, axis=0)

        print('grp shape', grp_array.shape)

        y_mtx = get_bitmap_from_shp(shapefile_path, rasterio.open(
            tifpath), os.path.join(tifpath[:-11], 'bitmap_WGS84.bmp'))

        if y_array == []:
            y_array = y_mtx.flatten()

        else:
            y_array = np.append(y_array, y_mtx.flatten(), axis=0)

    js.close()
    print('xarray shape', np.asarray(x_array).shape, y_array.shape)

    if shuffle:
        return unison_shuffled_copies(x_array, y_array)

    else:
        return x_array, y_array, y_mtx


def get_bitmap_from_shp(shp_path, rasterio_object, bitmap_path):

    geoms = []

    shapefile = fiona.open(shp_path)

    for shape in shapefile:
            # if sh['properties']['Start'] == start and sh['properties']['End'] == end:
        geoms.append(shape['geometry'])

    # raster the geoms onto a bitmap

    y_mtx = rasterio.features.rasterize(
        [(geo, 1) for geo in geoms],
        out_shape=(rasterio_object.shape[0], rasterio_object.shape[1]),
        transform=rasterio_object.transform)

    Image.fromarray(np.asarray(y_mtx * 255, dtype='uint8')).save(bitmap_path)

    rasterio_object.close()

    return y_mtx


def get_res_for_extent(extent, num_neighbor=5):

    full_extent = FULLDISK_EXTENT_COORDS
    full_res = FULL_RES_FD

    res = (int((extent[0] - extent[2]) * full_res[0] / (full_extent[0] - full_extent[2])),
           int((extent[1] - extent[3]) * full_res[1] / (full_extent[1] - full_extent[3])))

    res = (res[0] / num_neighbor * num_neighbor, res[1] / num_neighbor * num_neighbor)

    return res


def create_tile_pixel_out(img, tile_size=(5, 5), offset=(5, 5)):
    """
    create tiles of given image
    """

    img_shape = img.shape
    images = []

    # TODO: pixel unused.
    pixel = []

    for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
            # TODO: edge_size undefined
            pix = img[int(i + edge_size / 2), int(j + edge_size / 2)]
            # Debugging the tiles
            images.append(pix)
            # cv2.imwrite("debug_" + str(i) + "_" + str(j) + ".png", cropped_img)

    return images


def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a,b
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    return a[p], b[p]


# /*
#     img = Image.open(bmpfile)
#     x, y = img.shape()

#     grid_x = int(float(x) / grid_ratio)
#     grid_y = int(float(y) / grid_ratio)

#     grid_bool = np.zeros(grid_x, grid_y)

#     for i in range(grid_x):
#         for j in range(grid_y):
#             */


# def convert_bmp_to_coord(bmpfile, coverage_thres=0.5, grid_ratio=0.05):
#     """
#     desc : get bounding box from bmp image with black/white pixels.

#     """
