# -*- coding: utf-8 -*-
"""
Created on 15 nov 15:34:06 2018

@author: Karthick
"""

from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import get_arrays_from_json

class PixelModel():
    def __init__(self,num_neighbor=5):

        self.num_neighbor = num_neighbor


        ##########################
        # Make the model
        ##########################
        visible = Input(shape=(num_neighbor,num_neighbor,8,))
        conv1   = Conv2D(4, kernel_size=2, activation='relu', padding = 'same')(visible)
        pool1   = MaxPooling2D(pool_size=(2, 2),padding = 'same')(conv1)
        conv2   = Conv2D(2, kernel_size=2, activation='relu', padding = 'same')(pool1)
        flatten = Flatten()(conv2)
        #flatten = Flatten()(visible)
        dense1  = Dense(32,activation='relu')(flatten)
        dense2  = Dense(16, activation='relu')(dense1)
        dense2  = Dropout(0.5)(dense2)
        output  = Dense(1, activation='sigmoid')(dense2)

        self.model = Model(inputs=visible, outputs=output)





    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)

        except:
            print('the model does not conform with the weights given')

    def train(self,jsonfile,num_epoch):

        x_train,y_train = get_arrays_from_json(jsonfile,self.num_neighbor)
            
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model.train(x_train,y_train,
            num_epoch = num_epoch,
            batch_size = 10000,
            validation_split = 0.2)


    def save_model(self,savepath):
        self.model.save(savepath)


    def load_data(io_array):
        train = []
        test = []

        return train, test