import os
import urllib

import cv2
import mercantile
import numpy as np
import rasterio
import requests
import xarray
from pyorbital import astronomy
from tqdm import tqdm

from config import BANDS_LIST
from .data_rasterizer import DataRasterizer
from .goes_tiler import tiles
from .shape_utils import bitmap_from_shp
from .translator import create_cogeo

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
        for item in tqdm(self.jsondict):

            ncpath = item['ncfile']
            nctime = item['nctime']
            extent = item['extent']
            shapefile_path = item['shp']
            # rio_tiles = self.generate_tiles(nctime, extent)
            rio_tiles = self.generate_tiles_netcdf(ncpath, nctime, extent)
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


    def increment_if_low(self, start_num, end_num):
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


    def calculate_extents(self, extent):
        start_x, start_y, end_x, end_y = calculate_tile_xy(extent)
        # increment start tile number if end tile number is lesser than start
        start_x, end_x = self.increment_if_low(start_x, end_x)
        start_y, end_y = self.increment_if_low(start_y, end_y)
        extent = calculate_new_bbox(start_x, start_y, end_x, end_y)
        height = (end_x - start_x + 1) * TILE_SIZE
        width = (end_y - start_y + 1) * TILE_SIZE
        return start_x, start_y, end_x, end_y, width, height, extent


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

        rio_tiles_and_info = list()
        # format scene_id with the correct `m` string variation
        scene_id = SCENE_ID_FORMAT.format(
            M_STRING_VARIATIONS[int(int(nctime[:7]) > 2019092)],
            nctime,
        )

        start_x, start_y, end_x, end_y, width, height, extent = self.calculate_extents(
                extent
            )

        # loop through the tile range
        for y in range(start_y, end_y + 1):
            start_index_y = (y - start_y) * TILE_SIZE
            for x in range(start_x, end_x + 1):
                start_index_x = (x - start_x) * TILE_SIZE
                tile_url = WMTS_URL.format(ZOOM_LEVEL, x, y, scene_id)

                response = requests.get(tile_url)
                response.raise_for_status()
                if response.status_code == 200:
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


    def generate_tiles_netcdf(self, ncfile, nctime, extent, preprocess_flag=False):
        """generate wmts tiles from nctime time and extent (bounbing box)
        information and store it in self.save_path

        Args:
            nctime (str): netcdf namestring denoting time of scene
            extent (list): bounding box of the scene
            preprocess_flag (bool, optional): rad_to_ref convert flag

        Returns:
            TYPE: Description
        """
        nctime_long = nctime
        nctime = nctime[:9]  # subsetting nctime to remove minute strings
        rio_tiles_and_info = list()
        # format scene_id with the correct `m` string variation
        scene_id = SCENE_ID_FORMAT.format(
            M_STRING_VARIATIONS[int(int(nctime[:7]) > 2019092)],
            nctime_long,
        )

        start_x, start_y, end_x, end_y, width, height, extent = self.calculate_extents(
                extent
            )
        # Create a in-memory cogeo list from all netcdf bands
        raster_tif_list = list(map(
            create_cogeo,
            self.list_bands(
                ncfile, BANDS_LIST, nctime_long
            )
        ))
        # loop through the tile range
        # TODO: parallelize the double loop tiler
        for y in range(start_y, end_y + 1):
            start_index_y = (y - start_y) * TILE_SIZE
            for x in range(start_x, end_x + 1):
                start_index_x = (x - start_x) * TILE_SIZE
                tile, options = tiles(ZOOM_LEVEL, x, y, raster_tif_list, sceneid=scene_id)
                rio_tiff = rasterio.io.MemoryFile(tile).open()
                if tile:
                    if self.pre_process:
                        rio_data = self.rad_to_ref(rio_tiff, options)
                    else:
                        rio_data = rio_tiff.read()
                    rio_tiles_and_info.append(
                        (rio_data, rio_tiff.meta, f'{scene_id}_{y}_{x}')
                    )
                else:
                    with open('failed_links.txt', 'a') as file:
                        file.write(f"{ZOOM_LEVEL}, {x}, {y}")

        return rio_tiles_and_info


    def rad_to_ref(self, mem_file, auxillary_options, gamma=2.0):

        xarray_dataset = xarray.open_rasterio(mem_file)
        # move band axis to the end for vector multiplication
        # ignore last layer of xarray as it is ALPHA CHHANNEL
        rad = np.array(
            [band * float(auxillary_options[f'b{b_num}_kappa0']) for b_num, band in enumerate(
                xarray_dataset.data[:-1]
            )]
        )
        if self.cza_correct:
            x, y = np.meshgrid(xarray_dataset['x'], xarray_dataset['y'])
            cza = astronomy.cos_zen(float(auxillary_options['t']), x, y)
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
        '../data/train_val_list-v3.1.json',
        '../data/wmts_processed_train_val/',
        pre_process=True
    )
