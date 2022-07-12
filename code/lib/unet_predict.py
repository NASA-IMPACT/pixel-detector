from models import UNetModel
from keras.models import load_model
from data_preparer import UnetDataPreparer
from data_helper import unison_shuffled_copies
import json
from PIL import Image
import numpy as np

config = json.load(open('config.json'))

def predict(img_path):

    model = load_model(config['model_path'])
    dp = UnetDataPreparer(jsonfile='', save_path='../data/unet_images')
    x, y = dp.get_unet_data()
    x, y = unison_shuffled_copies(x, y)
    y = y > 128
    print(y[0])
    y = np.expand_dims(y, axis=-1)

    y_pred = model.predict(x, batch_size=10)

    for idx, item in enumerate(y_pred):
        Image.fromarray(item[:,:,0]*255).convert('L').save(str(idx)+'.bmp')