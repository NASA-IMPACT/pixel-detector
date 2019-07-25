# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-05-15 13:51:49
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-07-25 12:16:19

"""
Functions to handle feature shapes
"""

import fiona
import rasterio.features
import numpy as np


def bitmap_from_shp(shp_path, transform, img_shape):
    """ extract out the smoke pixels using the shapefile
     from the transform defined

    Args:
        shp_path (str): Shapefile path
        transfrom (rasterio.transfrom.Affine): rasterio transform object
    """
    geoms = []
    y_mtx = np.zeros((img_shape))
    with fiona.open(shp_path) as shapefile:

        for shape in shapefile:
            geoms.append(shape["geometry"])

        # raster the geoms onto a bitmap
    y_mtx = rasterio.features.rasterize(
        [(geo, 1) for geo in geoms],
        out_shape=(img_shape[0], img_shape[1]),
        transform=transform)

    return y_mtx
