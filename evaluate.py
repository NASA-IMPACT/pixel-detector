import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow
import keras
from PIL import Image
import json
from keras.layers import Input, Dense
from keras.models import load_model
from model import PixelModel as pix
from preprocessing import get_arrays_from_predict_json
import argparse
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix

argparser = argparse.ArgumentParser(
    description='Evaluate model on given json model')


argparser.add_argument(
    '-m',
    '--model',
    help='type of model')

argparser.add_argument(
    '-j',
    '--jsonfile',
    help='json file with ncfile and shapefile mapping')

argparser.add_argument(
    '-p',
    '--savepath',
    help='path of saved model')

argparser.add_argument(
    '-f',
    '--output',
    help='path of file')


def _main_(args):
    # build model
    jsonfile = args.jsonfile

    if args.savepath == None:
        print('give valid path to model')

    else:

        x,y,y_mtx = get_arrays_from_predict_json(jsonfile,5)
        print(x.shape,y.shape,y_mtx.shape)
        model = load_model(args.savepath)

        y_pred = model.predict(x,batch_size = 10000)
        y_mat = np.asarray(y_pred*255.0,dtype = 'uint8').reshape(y_mtx.shape)

        Image.fromarray(y_mat).save('test.bmp')


        fig = plt.figure(figsize=(12, 10), dpi=100)
        ax = fig.add_axes([0,0,1,1])
        plt.axis('off')
        ax.imshow(y_mtx, cmap='gray')
        temp_y_out = y_mat
        ax.imshow(temp_y_out, alpha=0.5, cmap='jet')
        plt.savefig(args.output)
        plt.close()

        # put confusion matrix text on image

        cm = confusion_matrix(y,y_pred>0.5)
        #cm = cm/float(y.size)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (y_mtx.shape[0] - 150,100)
        fontScale              = 0.50
        fontColor              = (0,0,0)
        lineType               = 2

        img = np.asarray(Image.open(args.output))

        cv2.putText(img,str(cm), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.imwrite(args.output,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

