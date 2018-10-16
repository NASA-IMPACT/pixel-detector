
from config import LANDSAT1,EGYPT_SHAPEFILE,NETHERLANDS_SHAPEFILE,OCEAN_SHAPEFILE,WGS84_DIR
from extract_features import extract_features
import numpy as np
from io_util import load_input,load_output
from sklearn.preprocessing import normalize


img_features1,bitmap_tiles1,img_shape = extract_features(LANDSAT3,64,NETHERLANDS_SHAPEFILE)

img_features2,bitmap_tiles2,img_shape2 = extract_features(LANDSAT2,64,EGYPT_SHAPEFILE)

inp1 = load_input(img_features1)
outp1 = load_output(bitmap_tiles1)

inp2 = load_input(img_features2)
outp2 = load_output(bitmap_tiles2)

rand_array1 =  np.hstack((inp1.T,outp1[:,None]))
rand_array2 =  np.hstack((inp2.T,outp2[:,None]))

rand_array = np.vstack((rand_array1,rand_array2))

ra = balanced_subsample(rand_array[:,:11],rand_array[:,11],subsample_size=1.0)

# rand_array = ra
x_train_t = rand_array[0:7200000,:11]
x_val_t = rand_array[7200000:7200000+2400000,:11]
x_test_t = rand_array[7200000+2400000:,:11]

y_train_t = rand_array[0:7200000,11]
y_val_t = rand_array[7200000:7200000+2400000,11]
y_test_t = rand_array[7200000+2400000:,11]
