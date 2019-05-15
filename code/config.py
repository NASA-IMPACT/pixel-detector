from os.path import join

# Directories which contain satellite imagery and shapefiles.
DATA_DIR = "/nas/rhome/mramasub/smoke_pixel_detector/data/"

# create a list of Bands here
BANDS_LIST = ['M3C01', 'M3C02', 'M3C03', 'M3C04', 'M3C05', 'M3C06', 'M3C07', 'M3C11']

# Directories to store everything related to the training data.
BITMAPS_DIR             = join('/cache/smoke_bitmaps/')
TIFF_DIR                = join(DATA_DIR, 'cache/WGS84_images/')
OUTPUT_DIR              = join(DATA_DIR, "output")
SCALE_FACTOR            = 1.0
PREDICT_THRESHOLD       = 0.4
CONUS_EXTENT_COORDS     = [-146.603349201, 14.561800658, -52.918301215, 56.001340454]
FULLDISK_EXTENT_COORDS  = [-149.765109632, -64.407370418, -0.234889169, 64.407370030]
FULL_RES                = [5497, 1713]
FULL_RES_FD             = [int(10848*SCALE_FACTOR), (10848*SCALE_FACTOR)]
BAND_1_FILENAME         = '0_WGS84.tif'
EVAL_DISP_STR           = 'A:{0:.2f}, R:{1:.2f}, P:{2:.2f}, IOU:{3:.2f}'
NUM_NEIGHBOR            = 0.5

