from tif_utils import reproject_dataset,reproject_dataset_no_transform, create_tiles, create_filter_tiles
import filters as flt
import shapefile_bitmap as sb
import json
from config import BANDS_LIST
from PIL import Image
import numpy as npy
from sklearn.utils import shuffle
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

#def extract_features(jsonfile,num_pixels_per_image=500,use_cache=False):
"""
    for the given RGB (TIFF) image and shapefile, generate a bitmap image for truth set and the
    concatenated feature set as npy array.
"""
jsonfile = 'custom_list.json'

with open(jsonfile, "r") as read_file:
    data = json.load(read_file)
num_bands = data[0]['image_paths'].__len__()

for item in data:
    # create bitmaps for each item  in data and use for classification
    raster_dataset, wgs_location = reproject_dataset(item['image_paths'][0])
    bitmap = sb.create_bitmap(raster_dataset, item['shapefile_path'], item['image_paths'][0])
    sat_arr = Image.open(satellite_path)
    sat_arr = np.array(sat_arr)
    num_posvals = sum(sum(bitmap==1))
    num_negvals = sum(sum(bitmap==0))
    X = np.concatenate([sat_arr[bitmap==1],sat_arr[bitmap==0]],axis=0)
    Y = np.concatenate([np.ones(num_posvals),np.zeros(num_negvals)],axis=0)
    Xs,Ys = shuffle(X,Y)
    Ys = one_hot(Ys.astype(int),2)
    
    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=1)
        
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    # This returns a tensor
    inputs = Input(shape=(3,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size = 1024,epochs=4,validation_data = (X_val,y_val))  # starts training
