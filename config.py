# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:06:21 2017

@author: Karthick
"""
import os
from os.path import join

# Directories which contain satellite imagery and shapefiles.
DATA_DIR = "/nas/rhome/mramasub/smoke_pixel_detector/data/"
# create list of Bands here
BANDS_LIST = ['M3C01','M3C02','M3C03','M3C04','M3C05','M3C06','M3C07','M3C11']
# Directories to store everything related to the training data.
BITMAPS_DIR = join('/cache/smoke_bitmaps/')
TIFF_DIR = join(DATA_DIR,'cache/WGS84_images/')
OUTPUT_DIR = join(DATA_DIR, "output")

# Code Variables

STORE_CACHE=False
NUM_PIXEL_PER_IMG=2000
