# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:34:06 2017

@author: Karthick
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

def compile_model(encoding_dim1, # this is the size of our encoded representations
                  encoding_dim2, # this is the size of our encoded representations
                 input,  # this is our input placeholder
                 output
                ):


    x_train = input[:,0:30000000]
    x_test = input[:,30000000:]

    input_img = Input(shape=(x_train.shape[0],))
    y_train = output[:30000000]
    y_test = output[30000000:]
    from keras.utils import np_utils

    #y_train = np_utils.to_categorical(y_train, 2)
    #y_test = np_utils.to_categorical(y_test, 1)
    #input = np.reshape(input)
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim1, activation='relu')(input_img)

    encoded2 = Dense(encoding_dim2, activation='softmax')(encoded)

    decoded = Dense(encoding_dim1, activation='relu')(encoded)

    decoded2 = Dense(input_img.shape[1]._value, activation='sigmoid')(decoded)

    autoencoder = Model(input=input_img,output=decoded)
    encoder = Model(input=input_img,output=encoded)
    #decoder = Model(input=input_img,output=decoded)

   # encoder = Model(input_img, encoded)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim1,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    encoder.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return autoencoder


def train_model(autoencoder,
                encoder,# this is the size of our encoded representations
                x_train,# this is our input placeholder
                x_test,
                y_train,
                y_test
                ):







    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(x_train.T, x_train.T,
                    epochs=1,
                    batch_size=25600,
                    shuffle=True,
                    validation_data=(x_test.T, x_test.T))

    encoder.fit(x_train.T, y_train,
              nb_epoch=3,
              batch_size=25600,
              shuffle=True,
              validation_data=(x_test.T, y_test))

    score = autoencoder.evaluate(x_test.T, x_test.T, verbose=1)

    score = encoder.evaluate(x_test[:30000,].T, y_test[:30000], verbose=1)
    print('Test score before fine turning:', score[0])
    print('Test accuracy after fine turning:', score[1])
    # encode and decode some digits
    # note that we take them from the *test* set
    #encoded_imgs = encoder.predict(x_test)
    #decoded_imgs = decoder.predict(encoded_imgs)

    # use Matplotlib (don't ask)




def load_data(io_array):
    train = []
    test = []

    return train, test