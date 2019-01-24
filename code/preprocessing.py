# @author Muthukumaran R.

import fiona
import json
import math
import numpy as np
import os
import rasterio
import rasterio.features
from itertools import product
# from config import shapefile_path,raw_data_path
from config import (
    BANDS_LIST,
    FULLDISK_EXTENT_COORDS,
    FULL_RES_FD
)
from PIL import Image
from tif_utils import create_array_from_nc, band_list
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from shapely import geometry
from shapely.ops import cascaded_union


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

            if i - half_edge < 0 or j - half_edge < 0 or i + half_edge + 1 > rows or
            j + half_edge + 1 > cols:
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

    # x_array = np.ndarray((0,0,num_neighbor,num_neighbor,8))
    # y_array = np.ndarray((0,0))
    x_array = []
    y_array = []
    cache_dir_list = []

    for item in jsondict:

        ncpath = item['ncfile']
        nctime = item['nctime']
        nclist = band_list(ncpath, BANDS_LIST, time=nctime)
        extent = item['extent']
        shapefile_path = item['shp']

        res = get_res_for_extent(extent, num_neighbor)
        ext_str = '_{}_{}_{}_{}'.format(extent[0], extent[1], extent[2], extent[3])
        arr, cache_dir = create_array_from_nc(nclist, fname=nctime + ext_str, extent=extent, res=res)

        grp_array = convert_pixels_to_groups(arr)
        cache_dir_list.append(cache_dir)
        x_array.append(np.asarray(grp_array))
        #append_to_list(x_array, grp_array)

        raster_object = rasterio.open(os.path.join(cache_dir,'0_WGS84.tif'))
        y_mtx = get_bitmap_from_shp(shapefile_path, raster_object,
            os.path.join(cache_dir, 'bitmap_WGS84.bmp'))
        raster_object.close()

        y_array.append(np.asarray(y_mtx.flatten()))
        #append_to_list(y_array, y_mtx.flatten())


    js.close()

    # if shuffle:
    #     return unison_shuffled_copies(x_array, y_array), cache_dir_list

    # else:
    return x_array, y_array, cache_dir_list




def append_to_list(lst, element):
    print('appending ', np.asarray(element).shape, 'to', lst.__len__())
    if lst != []:
        lst.append(element)
    else:
        lst = [element]
    print(lst)




def get_bitmap_from_shp(shp_path, rasterio_object, bitmap_path):

    geoms = []

    shapefile = fiona.open(shp_path)
    transform=rasterio_object.transform
    for shape in shapefile:
            # if sh['properties']['Start'] == start and sh['properties']['End'] == end:
        geoms.append(shape['geometry'])

    # raster the geoms onto a bitmap

    y_mtx = rasterio.features.rasterize(
        [(geo, 1) for geo in geoms],
        out_shape=(rasterio_object.shape[0], rasterio_object.shape[1]),
        transform=transform)

    Image.fromarray(np.asarray(y_mtx * 255, dtype='uint8')).save(bitmap_path)


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




def convert_xy_to_latlon(row, col, transform):
    """
    uses rasterio transform module to convert row, col of an image to
    its respective lat, lon coordinates
    """
    return rasterio.transform.xy(transform,row,col,offset='center')


def convert_bmp_to_shp(img_path, transform, shp_path, visualize_path=''):
    """
    Desc: make shapefile from white pixels of the BMP image
    """

    hull_points_list = get_hull_from_bmp(img_path)

    if visualize_path:
        vis_hull(img_path,visualize_path)

    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }
    with fiona.open(shp_path, 'w', 'ESRI Shapefile', schema) as output:

        for id_, points in enumerate(hull_points_list):
            poly = geometry.Polygon([convert_xy_to_latlon(x, y, transform) for x,y in points])

            output.write({
                    'geometry':geometry.mapping(poly),
                    'properties':{
                        'id':id_
                    },
                })




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





def get_hull_from_bmp(bmpfile,coverage_thres = 0.5, grid_ratio = 0.05):
    """
    desc : get bounding box from bmp image with black/white pixels.

    """

    im = np.asarray(Image.open(bmpfile).convert('L'))
    x,y = im.shape
    im = im.T
    row,col = np.where (im == 255)
    cluster_points = zip(row,col)


    clustering = DBSCAN(eps=2, min_samples=5).fit(cluster_points)
    labels = clustering.labels_
    print('clusters: ',set(labels).__len__())
    hull_points_list = list()

    for i in range(max(labels)+1):
        ith_cluster = [cluster_points[k] for k in np.where(labels == i)[0]]
        hull = ConvexHull(ith_cluster)
        hull_points_list.append([ith_cluster[k] for k in hull.vertices])

    return hull_points_list





def IOU_score(predicted_bmp, true_bmp):
    '''
    calculate IOU between the given bitmaps
    '''
    predict_hull = get_hull_from_bmp(predicted_bmp)
    true_hull    = get_hull_from_bmp(true_bmp)

    geom_intersection = cascaded_union(
        [geometry.Polygon(a).intersection(geometry.Polygon(b))
        for a, b in product(true_hull, predict_hull)])
    print(predict_hull)
    geom_union = cascaded_union([geometry.Polygon(a) for a in true_hull+predict_hull])


    return geom_intersection.area/geom_union.area




def vis_hull(bmpfile, outpath = 'test.png'):

    hull_points_list = get_hull_from_bmp(bmpfile)
    img = Image.open(bmpfile,'r')
    new_img = Image.new('RGBA', img.size)
    draw = ImageDraw.Draw(new_img, mode='RGBA')


    for i,pts in enumerate(hull_points_list):
        draw.polygon(pts, outline=(255,255,0))

    if outpath:
        img.save(outpath)



