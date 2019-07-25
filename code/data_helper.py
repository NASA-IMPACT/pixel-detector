# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-04-02 04:42:50
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-07-25 11:42:25

"""
Description: Helper methods for data generation
"""

from config import (
    BANDS_LIST,
    CACHE_DIR,
    SAT_H,
    SAT_SWEEP,
    SAT_LON,
)
from glob import glob
from PIL import Image
from rasterio_utils import wgs84_transform, generate_subsets
from shape_utils import smoke_pixels_from_shp

import cv2
import json
import numpy as np
import pickle
import scipy.ndimage
import sys
import xarray
import fiona
import os


def band_list(loc, band_array, time):
    """
    Get ncfile paths matching the bands and time from the 'loc' location
    """

    path_list = []

    print('checking for nc in ', loc)
    for band in band_array:
        fname = glob('{}/*{}*s{}*.nc'.format(loc, band, time))
        print('{}/*{}*s{}*.nc'.format(loc, band, time))
        print(fname)
        if fname == []:
            print('Nc Files not Found')
            sys.exit(1)
        else:
            path_list += fname

    return path_list


def extract_pixels(nclist, extent):
    """ extract pixels within given extent from raw files given in
        nclist.

    Args:
        nclist (list): list of ncfiles
        extent (list): subset extent (lat/lon bounds)
    """
    assert len(nclist) == len(BANDS_LIST)
    # res = lats_sub.shape
    res = GEO_RES
    array_list = []
    transforms = []
    for i, ncfile in enumerate(nclist):
        print('nclist content', ncfile)

        # read source dataset
        ds = xarray.open_dataset(str(ncfile), engine='h5netcdf')
        k = ds['kappa0'].data
        rad = ds['Rad'].data
        ref = np.clip(rad * k, 0, 1)
        gamma = 2.0
        if i == 3 or i == 5:  # if BAND_4 or BAND_6, then upsample
            ref = scipy.ndimage.zoom(ref, 2, order=0)
        if i == 1:  # if BAND_2 then downsample
            ref = rebin(ref, [res[0], res[1]])
        ref_255 = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)

        # reprojection to wgs84
        ref_255_wgs84, tf = wgs84_transform(ref_255, nclist[0], extent)
        # ref_255_wgs84 = histogram_equalize(ref_255_wgs84)
        print('reprojected band {} shape :'.format(i + 1), ref_255_wgs84.shape)
        array_list.append(ref_255_wgs84)
        transforms.append(tf)
        ds.close()

    return np.dstack(array_list), transforms


def get_data_unet(jsonfile, side_size=256):
    """Summary

    Args:
        jsonfile (json): json file containing the netcdf and shapefile
        paths
        num_neighbor (int, optional): num_neighbors to consider
    """
    print("reading from json file:", jsonfile)
    with open(jsonfile) as js:
        jsondict = json.loads(js.read())
        b_list = []
        x_list = []
        y_list = []
        lat_lon_list = []
        for item in jsondict:
            ncpath = item["ncfile"]
            nctime = item["nctime"]
            nclist = band_list(ncpath, BANDS_LIST, time=nctime)
            extent = item["extent"]
            shapefile_path = item["shp"]
            ext_str = "_{}_{}_{}_{}".format(
                extent[0], extent[1], extent[2], extent[3])
            cache_path = CACHE_DIR + '/' + nctime + '_' + ext_str + '.p'

            try:
                x_img = np.array(Image.open(os.path.join(cache_path, '.tiff')))
                y_img = np.array(Image.open(os.path.join(cache_path, '.bmp')))

            except IOError:  # file not found, create and store

                print('cannot find file, generating cache...')
                x_img, y_img = generate_images(nclist, shapefile_path, cache_path, side_size)

            if (x_img is not None) and (y_img is not None):
                x_list.append(x_img)
                y_list.append(y_img)

    return (x_list, y_list, b_list, lat_lon_list)


def generate_images(ncfiles, shapefile, cache_path, side_size=256):

    with fiona.open(shapefile) as shp:
        shp_center = ((shp.bounds[0] + shp.bounds[2]) / 2,
                      (shp.bounds[1] + shp.bounds[3]) / 2
                      )
        img_list = []
        for i, ncfile in enumerate(ncfiles):
            subsets, transforms = generate_subsets(
                ncfile, shp_center, os.path.join(cache_path, str(i)),
                side_size,
            )
            ds = xarray.open_dataset(str(ncfile), engine='h5netcdf')
            k = ds['kappa0'].data
            rad = ds['Rad'].data
            ref = np.clip(rad * k, 0, 1)
            gamma = 2.0

            if i == 3 or i == 5:  # if BAND_4 or BAND_6, then upsample
                ref = scipy.ndimage.zoom(ref, 2, order=0)
            if i == 1:  # if BAND_2 then downsample
                ref = rebin(ref, [res[0], res[1]])
            ref_255 = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)


def geo_idx(dd, dd_array):
    """
    search for nearest decimal degree in an array
    of decimal degrees and return the index.

    Args:
        dd (TYPE): Description
        dd_array (TYPE): Description

    Returns:
        TYPE: Description
    """
    geo_idx = (np.abs(dd_array - dd)).argmin()
    return geo_idx


def get_data(jsonfile, num_neighbor=5):
    """Summary

    Args:
        jsonfile (json): json file containing the netcdf and shapefile
        paths
        num_neighbor (int, optional): num_neighbors to consider
    """
    print("reading from json file:", jsonfile)
    with open(jsonfile) as js:
        jsondict = json.loads(js.read())
        b_list = []
        x_list = []
        y_list = []
        lat_lon_list = []
        for item in jsondict:
            ncpath = item["ncfile"]
            nctime = item["nctime"]
            nclist = band_list(ncpath, BANDS_LIST, time=nctime)
            extent = item["extent"]
            shapefile_path = item["shp"]
            ext_str = "_{}_{}_{}_{}".format(
                extent[0], extent[1], extent[2], extent[3])
            cache_path = CACHE_DIR + '/' + nctime + '_' + ext_str + '.p'

            try:
                (x_array_neighbors, y_array, x_array, lat_lon_grid) = pickle.load(
                    open(cache_path, 'rb'))
                b_list.append(x_array)
                lat_lon_list.append(lat_lon_grid)
                x_list.append(x_array_neighbors)
                y_list.append(y_array.flatten())

            except IOError:

                print('cannot find pickle file, generating cache...')
                x_array, transforms = extract_pixels(nclist, extent)
                y_array = smoke_pixels_from_shp(shapefile_path,
                                                transforms[0],
                                                x_array.shape[0:2])
                x_array_neighbors = convert_pixels_to_groups(x_array,
                                                             num_neighbor)
                # x_array_neighbors = 0
                # pickle.dump((x_array_neighbors,
                #              y_array, x_array, transforms),
                #             open(cache_path, 'wb'), protocol=3)
                b_list.append(x_array)
                lat_lon_list.append(transforms)
                x_list.append(x_array_neighbors)
                y_list.append(y_array.flatten())

    return (x_list, y_list, b_list, lat_lon_list)


def convert_pixels_to_groups(img, edge_size=5):
    """
    Given img[x,y] array, the method yields x*y arrays of edge_size*edge_size
    matrices, each array in output corresponds to each pixel in input
    """
    rows, cols, bands = img.shape
    print("Img shape", rows, cols, bands)
    result = np.zeros((rows * cols, edge_size, edge_size, bands))
    half_edge = int(edge_size / 2)
    idx = -1
    for i in range(rows):
        for j in range(cols):
            idx += 1
            moving_window = np.zeros(
                (edge_size, edge_size, bands), dtype=float)
            moving_window[half_edge, half_edge] = img[i, j]

            condition = i - half_edge < 0
            condition = condition or j - half_edge < 0
            condition = condition or i + half_edge + 1 > rows
            condition = condition or j + half_edge + 1 > cols

            if condition:
                moving_window[half_edge, half_edge] = img[i, j]
                result[idx] = moving_window

            else:
                result[idx] = img[i - half_edge:i + half_edge + 1,
                                  j - half_edge:j + half_edge + 1,
                                  ]

    return np.array(result)


def rebin(a, shape):
    """ rebins band 2 to 1KM resolution

    Args:
        a (TYPE): Description
        shape (TYPE): Description

    Returns:
        TYPE: Description
    """
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a,b

    Args:
        a (list/array): data a
        b (list/array): data a

    Returns:
        TYPE: a,b shuffled and resampled
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    # return balanced_subsample(a[p], b[p])
    return a[p], b[p]


def histogram_equalize(img):
    # return img
    return cv2.equalizeHist(img.astype('uint8'))


def balanced_subsample(x, y, subsample_size=1.00):
    """ subsample x,y such that they have sample ratio represented by
    subsample ratio

    Args:
        x (list/array): x data
        y (list/array): y data
        subsample_size (float, optional): subsample ratio (max/min)

    Returns:
        TYPE: Description
    """
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems is None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys
