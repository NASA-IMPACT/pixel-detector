
from config import LANDSAT1,EGYPT_SHAPEFILE,WGS84_DIR
from extract_features import extract_features
import numpy as np
from io_util import load_input,load_output
from sklearn.preprocessing import normalize


img_features1,bitmap_tiles1 = extract_features(LANDSAT1,64,EGYPT_SHAPEFILE)


inp = load_input(img_features1)


outp = load_output(bitmap_tiles1)

#inp_n = normalize(inp.T,axis=0)

np.save('inputs_11.npy',inp)
np.save('bitmap.npy',outp)
