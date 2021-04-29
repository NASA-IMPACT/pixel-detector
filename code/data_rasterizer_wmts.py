import cv2
import mercantile
import numpy as np
import os
import rasterio
import requests
import urllib
import xarray

from pyorbital import astronomy
from data_rasterizer import DataRasterizer
from shape_utils import bitmap_from_shp


TILE_SIZE = 256
WMTS_BASE_URL = "https://ffasjnxf3l.execute-api.us-east-1.amazonaws.com/production/"
WMTS_URL = f"{WMTS_BASE_URL}epsg4326/{{}}/{{}}/{{}}.tif?sceneid={{}}"
ZOOM_LEVEL = 7
SCENE_ID_FORMAT = 'OR_ABI-L1b-RadF-M{}_G16_s{}'
M_STRING_VARIATIONS = [3, 6]

KAPPA0 = [
    0.0015810000477358699,
    0.001954900100827217,
    0.003332100110128522,
    0.008836500346660614,
    0.013148699887096882,
    0.04147079959511757,
    0,
]


class DataRasterizerWmts(DataRasterizer):
    def __init__(self, jsonfile, save_path, pre_process):
        self.pre_process = pre_process
        self.cza_correct = False
        super(DataRasterizerWmts, self).__init__(
            jsonfile=jsonfile,
            save_path=save_path,
            cza_correct=self.cza_correct
        )

    def prepare_data(self):
        count = 0
        for item in self.jsondict:

            ncpath = item['ncfile']
            nctime = item['nctime']
            extent = item['extent']
            shapefile_path = item['shp']
            rio_tiles = self.generate_tiles(nctime, extent)
            for rio_data, rio_meta, scene_id in rio_tiles:

                # save tiffs
                tif_path = os.path.join(self.save_path, scene_id + '.tif')
                with rasterio.open(tif_path, 'w', **rio_meta) as dest:
                    for band_num, band in enumerate(rio_data):
                        dest.write(band, band_num + 1)
                count += 1

                # save labels
                bitmap_array = bitmap_from_shp(
                    shapefile_path,
                    rio_meta['transform'],
                    (rio_meta['width'], rio_meta['height'])
                )
                bmp_path = os.path.join(self.save_path, scene_id + '.bmp')
                self.save_image(bitmap_array.astype('uint8') * 255, bmp_path)
                print(f'processed and saved: {scene_id}')
        print(f'total tiles saved: {count}')

    def generate_tiles(self, nctime, extent, preprocess_flag=False):
        """generate wmts tiles from nctime time and extent (bounbing box)
        information and store it in self.save_path

        Args:
            nctime (str): netcdf namestring denoting time of scene
            extent (list): bounding box of the scene
            preprocess_flag (bool, optional): raf_to_ref convert flag

        Returns:
            TYPE: Description
        """
        nctime = nctime[:9]  # subsetting nctime to remove minute strings
        def increment_if_low(start_num, end_num):
            """ increment start tile number if end tile number is lesser than
            start

            Args:
                start_num
                end_num

            Returns:
                start_num, end_num
            """
            if start_num > end_num:
                start_num = end_num
            return start_num, end_num

        rio_tiles_and_info = list()
        # format scene_id with the correct `m` string variation
        scene_id = SCENE_ID_FORMAT.format(
            M_STRING_VARIATIONS[int(int(nctime[:7]) > 2019092)],
            nctime,
        )

        start_x, start_y, end_x, end_y = calculate_tile_xy(extent)
        # increment start tile number if end tile number is lesser than start
        start_x, end_x = increment_if_low(start_x, end_x)
        start_y, end_y = increment_if_low(start_y, end_y)
        extent = calculate_new_bbox(start_x, start_y, end_x, end_y)
        height = (end_x - start_x + 1) * TILE_SIZE
        width = (end_y - start_y + 1) * TILE_SIZE

        transform = rasterio.transform.from_bounds(
            *extent, height, width
        )

        # loop through the tile range
        for y in range(start_y, end_y + 1):
            start_index_y = (y - start_y) * TILE_SIZE
            for x in range(start_x, end_x + 1):
                start_index_x = (x - start_x) * TILE_SIZE
                tile_url = WMTS_URL.format(ZOOM_LEVEL, x, y, scene_id)

                response = requests.get(tile_url)
                if response.status_code != 500:
                    rio_tiff = rasterio.io.MemoryFile(response.content).open()
                    if self.pre_process:
                        rio_data = self.rad_to_ref(rio_tiff, nctime)
                    else:
                        rio_data = rio_tiff.read()
                    rio_meta = rio_tiff.meta
                    rio_tiles_and_info.append(
                        (rio_data, rio_tiff.meta, f'{scene_id}_{y}_{x}')
                    )
                else:
                    with open('failed_links.txt', 'a') as file:
                        file.write(tile_url)

        return rio_tiles_and_info

    def rad_to_ref(self, mem_file, nctime, gamma=2.0):

        # utc_time = convert_to_utc_format(nctime)
        xarray_dataset = xarray.open_rasterio(mem_file)

        # move band axis to the end for vector multiplication
        rad = np.array(
            [band * KAPPA0[b_num] for b_num, band in enumerate(
                xarray_dataset.data
            )]
        )
        if self.cza_correct:
            x, y = np.meshgrid(xarray_dataset['x'], xarray_dataset['y'])
            cza = astronomy.cos_zen(utc_time, x, y)
            rad = rad * cza
        ref = np.clip(rad, 0, 1)
        ref_clipped = np.floor(np.power(ref * 100, 1 / gamma) * 25.5)

        # move band axis back to leading axis for rasterio format
        return ref_clipped


def calculate_new_bbox(start_x, start_y, end_x, end_y):
    start_lon, _, _, end_lat = mercantile.bounds(
        start_x, start_y, ZOOM_LEVEL
    )
    _, start_lat, end_lon, _ = mercantile.bounds(
        end_x, end_y, ZOOM_LEVEL
    )
    # print(start_lon, start_lat, end_lon, end_lat)
    return [start_lon, start_lat, end_lon, end_lat]


def calculate_tile_xy(extent):
    start_x, start_y, _ = mercantile.tile(
        extent[0],
        extent[3],
        ZOOM_LEVEL
    )
    end_x, end_y, _ = mercantile.tile(extent[2], extent[1], ZOOM_LEVEL)
    return [start_x, start_y, end_x, end_y]


if __name__ == '__main__':
    drw = DataRasterizerWmts(
        'test_unet.json',
        '../wmts_processed_251/',
        pre_process=True
    )
