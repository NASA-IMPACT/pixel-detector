# @author Muthukumaran R.

import fiona
import geopandas
import json
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

from itertools import product
from PIL import Image
from scipy.spatial import ConvexHull
from shapely import geometry
from shapely.ops import cascaded_union
from sklearn.cluster import DBSCAN
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

            moving_window = np.zeros(
                (edge_size, edge_size, num_bands), dtype=float)
            moving_window[half_edge + 1, half_edge + 1] = img[i, j]

            condition = i - half_edge < 0
            condition = condition or j - half_edge < 0
            condition = condition or i + half_edge + 1 > rows
            condition = condition or j + half_edge + 1 > cols

            if condition:
                moving_window[half_edge + 1, half_edge + 1] = img[i, j]
                result.append(np.array(moving_window, dtype=float))

            else:
                result.append(np.array(img[i - half_edge:i + half_edge + 1,
                                           j - half_edge:j + half_edge + 1], dtype=float))

    return np.array(result)


def get_arrays_from_json(jsonfile, num_neighbor):
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

    print("reading json from :", jsonfile)

    js = open(jsonfile)
    jsondict = json.loads(js.read())

    # x_array = np.ndarray((0,0,num_neighbor,num_neighbor,8))
    # y_array = np.ndarray((0,0))
    x_array = []
    y_array = []
    cache_list = []

    for item in jsondict:

        ncpath = item["ncfile"]
        nctime = item["nctime"]
        nclist = band_list(ncpath, BANDS_LIST, time=nctime)
        extent = item["extent"]
        shapefile_path = item["shp"]

        res = get_res_for_extent(extent, num_neighbor)
        ext_str = "_{}_{}_{}_{}".format(extent[0], extent[1], extent[2], extent[3])

        # workflow for train and evaluate
        arr, cache_dir = create_array_from_nc(
            nclist,
            fname=nctime + ext_str,
            extent=extent,
            res=res,
        )
        cache_list.append(cache_dir)
        grp_array = convert_pixels_to_groups(arr)
        x_array.append(np.asarray(grp_array))
        #append_to_list(x_array, grp_array)

        with rasterio.open(os.path.join(cache_dir, "0_WGS84.tif")) as raster_object:
            y_mtx = get_bitmap_from_shp(shapefile_path,
                                        raster_object,
                                        os.path.join(cache_dir, "bitmap_WGS84.bmp"))

            y_array.append(np.asarray(y_mtx.flatten()))

    js.close()

    return x_array, y_array, cache_list


def get_arrays_for_prediction(jsondict, num_neighbor, create_cache=False):
    """
    num_neighbor = number of pixels at the egde of the square matrix input
    jsonfile = {
      "ncfile": "<ppath/to/ncfile>",
      "nctime": "unique time string of the ncfile",
      "extent": [extent coordinates in lat/lon],
    }

    """
    # workflow for testing

    print("item length:", len(jsondict))

    # x_array = np.ndarray((0,0,num_neighbor,num_neighbor,8))
    # y_array = np.ndarray((0,0))

    x_array = []
    auxillary_list = []

    for item in jsondict:

        ncpath = item["ncfile"]
        nctime = item["nctime"]
        nclist = band_list(ncpath, BANDS_LIST, time=nctime)
        extent = item["extent"]
        res = get_res_for_extent(extent, num_neighbor)

        arr, raster_transform = create_array_from_nc(
            nclist, fname="", extent=extent, res=res)
        auxillary = (raster_transform, res)
        grp_array = convert_pixels_to_groups(arr)

        auxillary_list.append(auxillary)
        x_array.append(np.asarray(grp_array))

    return x_array, auxillary_list


def append_to_list(lst, element):

    print("appending ", np.asarray(element).shape, "to", lst.__len__())
    if lst != []:
        lst.append(element)
    else:
        lst = [element]
    print(lst)


def get_bitmap_from_shp(shp_path, rasterio_object, bitmap_path):

    geoms = []

    shapefile = fiona.open(shp_path)
    transform = rasterio_object.transform
    for shape in shapefile:
            # if sh["properties"]["Start"] == start and sh["properties"]["End"] == end:
        geoms.append(shape["geometry"])

    # raster the geoms onto a bitmap

    try:
        y_mtx = rasterio.features.rasterize(
            [(geo, 1) for geo in geoms],
            out_shape=(rasterio_object.shape[0], rasterio_object.shape[1]),
            transform=transform
            )

    except:
        print("no objects found")
        y_mtx = np.zeros((rasterio_object.shape[0], rasterio_object.shape[1]))

    Image.fromarray(np.asarray(y_mtx * 255, dtype="uint8")).save(bitmap_path)

    return y_mtx


def get_res_for_extent(extent, num_neighbor=5):

    full_extent = FULLDISK_EXTENT_COORDS
    full_res = FULL_RES_FD

    res = (int((extent[0] - extent[2]) * full_res[0] / (full_extent[0] - full_extent[2])),
           int((extent[1] - extent[3]) * full_res[1] / (full_extent[1] - full_extent[3])))

    res = (res[0] / num_neighbor * num_neighbor,
           res[1] / num_neighbor * num_neighbor)

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
    return rasterio.transform.xy(transform, col, row, offset="center")


def convert_bmp_to_shp(img, transform, shp_path, visualize_path=""):
    """
    Desc: make shapefile from white pixels of the BMP image
    """

    hull_points_list = get_hull_from_bmp(img)

    if not hull_points_list:

        print("No shapes found, No shapefile will be generated")

    if visualize_path:
        vis_hull(img, visualize_path)

    schema = {
        "geometry": "Polygon",
        "properties": {"id": "int"},
    }
    if shp_path:
        with fiona.open(shp_path, "w", "ESRI Shapefile", schema) as output:

            for id_, points in enumerate(hull_points_list):
                poly = geometry.Polygon(
                    [convert_xy_to_latlon(x, y, transform) for x, y in points])

                output.write({
                    "geometry":     geometry.mapping(poly),
                    "properties":   {"id": id_},
                })
    else:

        geojson_dict = []

        for id_, points in enumerate(hull_points_list):
            poly = geometry.Polygon(
                [convert_xy_to_latlon(x, y, transform) for x, y in points])

            pol_dict = {
                'geometry':     geopandas.GeoSeries([poly]).to_json(),
                'properties':   {'id': id_}
            }

            geojson_dict.append(pol_dict)

        geo_collection = {
            'type':     'FeatureCollection',
                        'features': geojson_dict,
        }

        return geo_collection


def get_res_for_extent(extent, num_neighbor=5):

    full_extent = FULLDISK_EXTENT_COORDS
    full_res = FULL_RES_FD

    res = (int((extent[0] - extent[2]) * full_res[0] / (full_extent[0] - full_extent[2])),
           int((extent[1] - extent[3]) * full_res[1] / (full_extent[1] - full_extent[3])))

    res = (res[0] / num_neighbor * num_neighbor,
           res[1] / num_neighbor * num_neighbor)

    return res


def create_tile_pixel_out(img, tile_size=(5, 5), offset=(5, 5)):
    """
    create tiles of given image
    """

    img_shape = img.shape
    images = []
    pixel = []

    for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
            pix = img[int(i + edge_size / 2), int(j + edge_size / 2)]
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


def get_hull_from_bmp(img, coverage_thres=0.5, grid_ratio=0.05):
    """
    desc : get bounding box from bmp image with black/white pixels.

    """

    im = np.asarray(img)
    x, y = im.shape
    im = im.T
    row, col = np.where(im == 255)
    cluster_points = zip(row, col)

    hull_points_list = list()

    if cluster_points:

        print("Clustering smoke plumes using DBSCAN...")
        clustering = DBSCAN(eps=2, min_samples=5).fit(cluster_points)
        labels = clustering.labels_
        #print("clusters: ",set(labels).__len__())

        for i in range(max(labels) + 1):

            ith_cluster = [cluster_points[k] for k in np.where(labels == i)[0]]

            # Check if clusters are not in a single line
            if len(set(np.asarray(ith_cluster).T[0])) > 1\
                    and len(set(np.asarray(ith_cluster).T[1])) > 1:

                hull = ConvexHull(ith_cluster)
                hull_points_list.append([ith_cluster[k]
                                         for k in hull.vertices])

    return hull_points_list


def IOU_score(predicted_bmp, true_bmp):
    """
    calculate IOU between the given bitmaps
    """

    predict_hull = get_hull_from_bmp(predicted_bmp)
    true_hull = get_hull_from_bmp(true_bmp)

    geom_intersection = cascaded_union(
        [geometry.Polygon(a).intersection(geometry.Polygon(b))
         for a, b in product(true_hull, predict_hull)])
    geom_union = cascaded_union([geometry.Polygon(a)
                                 for a in true_hull + predict_hull])

    return geom_intersection.area / geom_union.area


def vis_hull(bmpfile, outpath="test.png"):

    hull_points_list = get_hull_from_bmp(bmpfile)
    img = Image.open(bmpfile, "r")
    new_img = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(new_img, mode="RGBA")

    for i, pts in enumerate(hull_points_list):
        draw.polygon(pts, outline=(255, 255, 0))

    if outpath:
        img.save(outpath)
