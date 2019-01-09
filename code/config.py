from os.path import join

# Directories which contain satellite imagery and shapefiles.
DATA_DIR = "/nas/rhome/mramasub/smoke_pixel_detector/data/"

# create a list of Bands here
BANDS_LIST = ['M3C01', 'M3C02', 'M3C03', 'M3C04', 'M3C05', 'M3C06', 'M3C07', 'M3C11']
# BANDS_LIST = ['M4C01','M3C02','M4C03','M4C04','M4C05','M4C06','M4C07','M4C11']

# Directories to store everything related to the training data.
BITMAPS_DIR = join('/cache/smoke_bitmaps/')
TIFF_DIR = join(DATA_DIR, 'cache/WGS84_images/')
OUTPUT_DIR = join(DATA_DIR, "output")

# Constants
STORE_CACHE = False
NUM_PIXEL_PER_IMG = 2000
PREDICT_THRESHOLD = 0.5
CONUS_EXTENT_COORDS = [-146.603349201, 14.561800658, -52.918301215, 56.001340454]
FULLDISK_EXTENT_COORDS = [-149.765109632, -64.407370418, -0.234889169, 64.407370030]
FULL_RES = [5497, 1713]
FULL_RES_FD = [10848, 10848]
