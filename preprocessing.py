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

def convert_pixels_to_groups(X,edge_size=5,stride=0,num_bands = 8):
    
    rows,cols = X.shape[0],X.shape[1]
    mrows = int((1+(rows/edge_size)) * edge_size)
    mcols = int((1+(cols/edge_size)) * edge_size)
    
    #num_bands = X.shape[2]
    result  = []
    half_edge = int(edge_size/2)
    if stride >0 :
        raise notImplementedError

    for i in range(rows):
        for j in range(cols):
            if i - half_edge < 0 or j - half_edge <0 or i + half_edge+1>rows or j + half_edge +1> cols:
                if num_bands == 1:
                    m = np.zeros((edge_size,edge_size),dtype=float)
                else:
                    m = np.zeros((edge_size,edge_size,num_bands),dtype=float)
                m[half_edge+1,half_edge+1] = X[i,j]
                result.append(np.array(m, dtype=float))

            else:
                result.append(np.array(X[i-half_edge:i+half_edge+1,j-half_edge:j+half_edge+1], dtype=float))
    return np.array(result)


def get_arrays_from_json(jsonfile,num_neighbor):

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
                
        b1_raster = rasterio.open(tifpath)

        print(tifpath,geoms)

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
    return unison_shuffled_copies(x_array,y_array)


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


def create_tiles(img):

    img_shape = img.shape
    tile_size = (5, 5)
    offset = (5, 5)
    images = []
    
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            # Debugging the tiles
            images.append(cropped_img)
            #cv2.imwrite("debug_" + str(i) + "_" + str(j) + ".png", cropped_img)
    
    return images

def create_tile_pixel_out(img,tile_size = (5,5),offset = (5, 5)):

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
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    return a[p], b[p]



def balanced_sample_maker(X, y, sample_size, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train=X[balanced_copy_idx]
    labels_train=y[balanced_copy_idx]
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels, values = zip(*Counter(labels_train).items())
    print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    print(check)
    if check == True:
        print('Good all classes have the same number of examples')
    else:
        print('Repeat again your sampling your classes are not balanced')
    indexes = np.arange(len(labels))
    width = 0.5
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()
    return data_train,labels_train
