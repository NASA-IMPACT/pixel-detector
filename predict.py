from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from PIL import Image

def predict_overlay(sat_image,model,bmp_img_path= 'test.png'):
    sat_img = Image.open(satellite_path)
    sat_arr = np.array(sat_img)
    pred_array = model.predict(sat_arr.reshape(sat_arr.shape[0]*sat_arr.shape[1],sat_arr.shape[2]),batch_size=1024)
    
    pred_bmp = np.int8(pred_array[:,1]>0.5)*255
    img = Image.fromarray(pred_bmp.reshape(sat_arr.shape[0],sat_arr.shape[1],order='C'), mode = 'L')
    img.save('my.png')

    background = sat_img.convert("RGBA")
    overlay = img.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.5)
    new_img.save(bmp_img_path,"PNG")


