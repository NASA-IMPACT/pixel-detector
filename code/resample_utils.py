from rasterio.warp import (
    calculate_default_transform,
    reproject,
)
from rasterio.io import MemoryFile

import rasterio


def reproject_ncfile(ncifle_path, extent, projection='epsg:4326'):
    """reproject the given ncfile to 'projection'

    Args:
        ncifle (TYPE): path/to/ncfile
        extent (TYPE): extent to work with
        projection (str, optional): projection type
    """

    with rasterio.open(ncifle_path) as dataset:
        meta = dataset.profile
        new_tf, width, height = calculate_default_transform(
            dataset.crs,
            projection,
            dataset.width,
            dataset.height,
            *dataset.bounds
        )

        meta.update(
            driver='GTiff',
            crs={'init': projection},
            transform=new_tf,
            nodata=0,
        )

        with MemoryFile() as temp_file:
            with rasterio.open(temp_file, 'w', **meta) as dst:
                reproject(
                    source=rasterio.band(dataset, 1),
                    destination=rasterio.band(dst, 1)
                )
