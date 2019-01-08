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
from preprocessing import get_arrays_from_json
import argparse
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
from config import PREDICT_THRESHOLD


class Evaluate:

    def __init__(self, config):
        """
        config = {
            'type': < choice of 'pixel' or 'deconv'>,
            'num_neighbor': < TODO: explain this parameter >,
            ''
        }
    """

        self.config = config
        #self.model_holder = MODELS[self.config['type']] (self.config)

    def evaluate(self):

        x,y,y_mtx = get_arrays_from_json(self.config['eval_json'],5, shuffle=False)
        print(x.shape,y.shape,y_mtx.shape)
        model = load_model(str(self.config['savepath']))

        y_pred = model.predict(x,batch_size = 10000)
        y_pred = y_pred > PREDICT_THRESHOLD
        y_mat = np.asarray(y_pred*255.0,dtype = 'uint8').reshape(y_mtx.shape)

        Image.fromarray(y_mat).save('eval2.bmp')


        fig = plt.figure(figsize=(25, 15), dpi=100)
        ax = fig.add_axes([0,0,1,1])
        plt.axis('off')
        ax.imshow(y_mtx, cmap='gray')
        temp_y_out = y_mat
        ax.imshow(temp_y_out, alpha=0.5, cmap='jet')
        plt.savefig('eval2.png')
        plt.close()

        #put confusion matrix text on image

        cm = confusion_matrix(y,y_pred>PREDICT_THRESHOLD)
        #cm = cm/float(y.size)
        acc = float(cm[0][0] + cm[1][1]) / float(cm[0][0]+cm[0][1]+cm[1][1] + cm[1][0])

        recall = float(cm[1][1]) / float(cm[1][0] +cm [1][1])

        precision = float(cm[1][1]) / float(cm[0][1]+cm[1][1])

        disp_str = 'a:{0:.2f},r:{1:.2f},p:{2:.2f}'.format(acc,recall,precision)
        #disp_str = 'null'
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100, 100)
        fontScale              = 0.50
        fontColor              = (255,255,255)
        lineType               = 2

        img = np.asarray(Image.open('eval.png'))

        print('arc:',disp_str)
        cv2.putText(img,disp_str,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

        cv2.imwrite('eval.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # build model

    eval = Evaluate(json.load(open('config.json')))
    eval.evaluate()





