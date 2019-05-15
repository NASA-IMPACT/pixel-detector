# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-04-02 04:40:19
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-05-15 13:58:08

SCALE_FACTOR = 1.0
PREDICT_THRESHOLD = 0.50
BANDS_LIST = ['M3C01', 'M3C02', 'M3C03']

# satellite information constants
SAT_H = 35786023.0
SAT_LON = -75.0
SAT_SWEEP = 'x'
GEO_RES = (10848, 10848)
LAT_LON_IDX = '../data/lat_lon_reprojected_py3.pkl'
OUTPUT_DIR = '../data/eval_outputs/'
CACHE_DIR = '../data/cache'
