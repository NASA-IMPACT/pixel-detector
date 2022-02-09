# Model Constants
SCALE_FACTOR = 1.0
IMG_SCALE = 255
BANDS_LIST = ['01', '02', '03', '04', '05', '06']

# Satellite information constants
SAT_H = 35786023.0
SAT_LON = -75.0
SAT_SWEEP = 'x'

# Directories
LAT_LON_IDX = '../data/lat_lon_reprojected_py3.pkl'
OUTPUT_DIR = '../data/eval_outputs_smoke_no_cza/'
CACHE_DIR = '../data/cache'

# Infer-visualization parameters
THRESHOLD = 0.1
MODEL_PATH = f'../models/smoke_wmts_ref.h5'
MODEL_INPUT_DATA_DIR = f'../data/wmts_processed'
MODEL_OUTPUT_DIR = f'../data/smoke_wmts_ref_preds_t{THRESHOLD}'
TEMP_DIR = f'../data/temp'
