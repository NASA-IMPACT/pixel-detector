# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:06:21 2017

@author: Karthick
"""
import os
from os.path import join

# Directories which contain satellite imagery and shapefiles.
DATA_DIR = "/nas/rgroup/dsig/dataset/smoke/Terra"
IMG_DIR = DATA_DIR
# create list of Bands here
BANDS_LIST = ['M3C01','M3C02','M3C03','M3C04','M3C05','M3C06','M3C07','M3C11']

# SHAPEFILE CONFIG #
#SHAPEFILE_DIR = join(DATA_DIR, "input", "shapefiles")
SHAPEFILE_DIR='/nas/rhome/mramasub/smoke_pixel_detector/data/input/shapefiles/2016'
# create shapefiles corresponding to images here

#SHAPEFILE_DIR = join(SHAPEFILE_DIR, "2016")

# Directories to store everything related to the training data.
TRAIN_DATA_DIR = join(DATA_DIR, "cache", "train_data")
TILES_DIR = join(TRAIN_DATA_DIR, "tiles")
BITMAPS_DIR = '/nas/rhome/mramasub/smoke_pixel_detector/data/cache/smoke_bitmaps/'
WGS84_DIR = '/nas/rhome/mramasub/smoke_pixel_detector/data/cache/WGS84_images/'
LABELS_DIR = join(TRAIN_DATA_DIR, "labels_images")
OUTPUT_DIR = join(DATA_DIR, "output")

# Code Variables

STORE_CACHE=False
NUM_PIXEL_PER_IMG=2000
