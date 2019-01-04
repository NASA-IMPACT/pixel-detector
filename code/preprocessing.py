# @author Muthukumaran R.

import fiona
from netCDF4 import Dataset
import numpy as np
#from config import shapefile_path,raw_data_path
from datetime import datetime
import os
from glob import glob
from datetime import timedelta
from osgeo import gdal, osr, gdal_array
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyresample import image, geometry, utils,kd_tree,bilinear
import rasterio, rasterio.features
from config import BANDS_LIST,CONUS_EXTENT_COORDS,FULL_RES,FULLDISK_EXTENT_COORDS,FULL_RES_FD
import math
from tif_utils import create_array_from_nc, band_list
import json
from PIL import Image

def convert_pixels_to_groups(img,edge_size=5,stride=0,num_bands = 8):
    """
    Given img[x,y] array, the method yields x*y arrays of edge_size*edge_size
    matrices, each array in output corresponds to each pixel in input
    """
    rows,cols,bands = img.shape
    result  = []
    half_edge = int(edge_size/2)
    # if stride > 0:
    #     raise notImplementedError

    for i in range(rows):
        for j in range(cols):
            moving_window = np.zeros((edge_size, edge_size,num_bands), dtype=float)
            moving_window[half_edge + 1, half_edge + 1] = img[i, j]
            row_extent = i if i - half_edge < 0 else 0
            height_extent = j if j - half_edge < 0 else 0

            if i - half_edge < 0 or j - half_edge < 0 or i + half_edge+1>rows or j + half_edge +1> cols:
                moving_window[half_edge+1,half_edge+1] = img[i, j]
                result.append(np.array(moving_window, dtype=float))

            else:
                result.append(np.array(img[i-half_edge:i+half_edge+1,j-half_edge:j+half_edge+1], dtype=float))

    return np.array(result)


def get_arrays_from_json(jsonfile,num_neighbor,shuffle=True):
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

    js = open(jsonfile)
    jsondict = json.loads(js.read())

    x_array = []
    y_array = []

    for item in jsondict:

        ncpath = item['ncfile']
        nctime = item['nctime']
        nclist = band_list(ncpath,BANDS_LIST,time = nctime)
        extent = item['extent']
        shapefile_path = item['shp']
        start = item['start']
        end = item['end']

        res = get_res_for_extent(extent,num_neighbor)
        ext_str = '_{}_{}_{}_{}'.format(extent[0],extent[1],extent[2],extent[3])
        arr,tifpath = create_array_from_nc(nclist,fname = nctime+ext_str,extent=extent,res=res)
        grp_array = convert_pixels_to_groups(arr)

        if x_array == []:
            x_array = grp_array

        else:
            x_array = np.append(x_array,grp_array,axis = 0)

        print('grp shape',grp_array.shape)

        y_mtx = get_bitmap_from_shp(shapefile_path,rasterio.open(tifpath),os.path.join(tifpath[:-11],'bitmap_WGS84.bmp'))

        if y_array == []:
            y_array = y_mtx.flatten()

        else:
            y_array = np.append(y_array,y_mtx.flatten(),axis = 0)

    js.close()
    print('xarray shape',np.asarray(x_array).shape,y_array.shape)

    if shuffle:
        return unison_shuffled_copies(x_array,y_array)

    else:
        return x_array,y_array



# hopefully donot use this. (if we can maintain the size of the arrays from get_arrays_from_json)

def get_arrays_from_predict_json(jsonfile,num_neighbor):

    js = open(jsonfile)
    jsondict = json.loads(js.read())

    x_array = []
    y_array = []

    for item in jsondict:
        ncpath = item['ncfile']
        nctime = item['nctime']
        nclist = band_list(ncpath,BANDS_LIST,time = nctime)
        extent = item['extent']
        shapefile = fiona.open(item['shp'])
        start = item['start']
        end = item['end']

        full_extent = FULLDISK_EXTENT_COORDS
        full_res = FULL_RES_FD

        res = (int((extent[0] - extent[2])*full_res[0] /(full_extent[0] - full_extent[2])),
            int((extent[1] - extent[3])*full_res[1] /(full_extent[1] - full_extent[3])))
        res = (res[0]/num_neighbor*num_neighbor,res[1]/num_neighbor*num_neighbor)

        ext_str = '_{}_{}_{}_{}'.format(extent[0],extent[1],extent[2],extent[3])
        arr,tifpath = create_array_from_nc(nclist,fname = nctime+ext_str,extent=extent,res=res)

        grp_array = convert_pixels_to_groups(arr)

        if x_array == []:
            x_array = grp_array

        else:
            x_array = np.append(x_array,grp_array,axis = 0)

        print('grp shape',grp_array.shape)

        geoms = []


        for sh in shapefile:
            if sh['properties']['Start'] == start and sh['properties']['End'] == end:
                geoms.append(sh['geometry'])

        print(tifpath,geoms)

        b1_raster = rasterio.open(tifpath)

        y_mtx = rasterio.features.rasterize(
                 [(geo,1) for geo in geoms],
                 out_shape=(b1_raster.shape[0],b1_raster.shape[1]),
                 transform=b1_raster.transform)

        if y_array == []:
            y_array = y_mtx.flatten()

        else:
            y_array = np.append(y_array,y_mtx.flatten(),axis = 0)

        b1_raster.close()

        Image.fromarray(np.asarray(y_mtx*255,dtype = 'uint8')).save(os.path.join(tifpath[:-11],'bitmap_WGS84.bmp'))

    js.close()
    print('xarray shape',np.asarray(x_array).shape,y_array.shape)
    return x_array,y_array,y_mtx*255.0






def get_bitmap_from_shp(shp_path, rasterio_object, bitmap_path):

        geoms = []

        shapefile = fiona.open(shp_path)

        for shape in shapefile:
            #if sh['properties']['Start'] == start and sh['properties']['End'] == end:
            geoms.append(shape['geometry'])

        # raster the geoms onto a bitmap

        y_mtx = rasterio.features.rasterize(
                 [(geo,1) for geo in geoms],
                 out_shape=(rasterio_object.shape[0],rasterio_object.shape[1]),
                 transform=rasterio_object.transform)

        Image.fromarray(np.asarray(y_mtx*255,dtype = 'uint8')).save(bitmap_path)

        rasterio_object.close()

        return y_mtx






def get_res_for_extent(extent, num_neighbor = 5):

    full_extent = FULLDISK_EXTENT_COORDS
    full_res = FULL_RES_FD

    res = (int((extent[0] - extent[2])*full_res[0] /(full_extent[0] - full_extent[2])),
        int((extent[1] - extent[3])*full_res[1] /(full_extent[1] - full_extent[3])))

    res = (res[0]/num_neighbor*num_neighbor,res[1]/num_neighbor*num_neighbor)

    return res

def create_tile_pixel_out(img,tile_size = (5,5),offset = (5, 5)):
    """
    create tiles of given image
    """

    img_shape = img.shape
    images = []
    pixel = []
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            pix = img[int(i+edge_size/2),int(j+edge_size/2)]
            # Debugging the tiles
            images.append(pix)
            #cv2.imwrite("debug_" + str(i) + "_" + str(j) + ".png", cropped_img)

    return images



def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a,b
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    return a[p], b[p]

