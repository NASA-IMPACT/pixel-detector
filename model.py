# -*- coding: utf-8 -*-
"""
Created on 15 nov 15:34:06 2018

@author: Karthick
"""

from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import get_arrays_from_json


class PixelModel():
    def __init__(self,num_neighbor=5):

        self.num_neighbor = num_neighbor

        self.make_model()


    def make_model(self):

        ##########################
        # Make the model
        ##########################
        
        visible = Input(shape=(self.num_neighbor,self.num_neighbor,8,))
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

        self.model.summary()       



    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)

        except:
            print('the model does not conform with the weights given')



    def train(self,jsonfile,num_epoch,savepath):

        x_train,y_train = get_arrays_from_json(jsonfile,self.num_neighbor)
            
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])



        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath=savepath, verbose=1, save_best_only=True)


        self.model.fit(x_train,y_train,
            nb_epoch = num_epoch,
            batch_size = 10000,
            callbacks=[early_stopping,checkpointer],
            validation_split = 0.2)


    def save_model(self,savepath):
        self.model.save(savepath)


    def load_data(io_array):
        train = []
        test = []

        return train, test


class DeconvModel():
    def __init__(self,num_neighbor=5):

        self.num_neighbor = num_neighbor

        self.make_model()


    def make_model(self):

        ##########################
        # Make the model
        ##########################
        visible = Input(shape=(self.num_neighbor,self.num_neighbor,8,))
        x = ZeroPadding2D(((2,1),(1,2)))(visible)
        x = Conv2D(10, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((3, 3), padding='same')(x)
        x = Conv2D(15, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)



        x = Conv2D(10, (2, 2), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(5, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((3, 3))(x)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Cropping2D(((2,2),(2,2)))(x)
        self.model  = Model(visible, x)
        self.model.summary()


    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)

        except:
            print('the model does not conform with the weights given')

    def train(self,jsonfile,num_epoch,savepath):

        x_train,y_train = get_arrays_from_json(jsonfile,self.num_neighbor)
            
        self.model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath=savepath, verbose=1, save_best_only=True)

        self.model.fit(x=X_train, y=np.expand_dims(Y_train,axis = -1), 
            batch_size=32, epochs=200, verbose=1,validation_split = 0.2,
            callbacks=[early_stopping,checkpointer],
            shuffle=True)


    def save_model(self,savepath):
        self.model.save(savepath)


