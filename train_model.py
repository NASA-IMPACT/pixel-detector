from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import keras

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

# y_train = np_utils.to_categorical(y_train, 2)
# y_test = np_utils.to_categorical(y_test, 1)
# input = np.reshape(input)
# "encoded" is the encoded representation of the input



input_img = Input(shape=(x_train.shape[1],))
full_encoder = Dense(5, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_img)
sparse_ae = Model(input=input_img,output=full_encoder)
sparse_ae.compile(optimizer='sgd',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

tbCallback = keras.callbacks.TensorBoard(log_dir='./',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

sparse_ae.fit(x_train_t,
                x_train_t,
                batch_size=250000,
                nb_epoch=10,
                callbacks=[tbCallback],
                validation_data=(x_val_t, x_val_t)
                )


input_img = Input(shape=(x_train.shape[1],))

encoder = Dense(8, activation='sigmoid')(input_img)

encoder2 = Dense(5, activation='sigmoid')(encoder)

decoder2 = Dense(8, activation='sigmoid')(encoder2)

decoder = Dense(11, activation='linear')(decoder2)

autoencoder = Model(input=input_img,output=decoder)

autoencoder.compile(optimizer='sgd',
                    loss='mean_squared_error',metrics=['accuracy'])



tbCallback = keras.callbacks.TensorBoard(log_dir='./',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

autoencoder.fit(x_train_t,
                x_train_t,
                batch_size=2500,
                nb_epoch=10,
                callbacks=[tbCallback],
                validation_data=(x_val_t, x_val_t)
                )

encoder_model = Model(input=input_img,output=encoder2)

x_train_5 = encoder_model.predict(x_train_t.T)

input_compressed = Input(shape=(x_train_5.shape[1],))

hidden_layer = Dense(3,activation='tanh')(input_compressed)

classify_layer = Dense(1, activation='sigmoid',
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros')(hidden_layer)
#autoencoder.test(x_test.T,x_test.T, batch_size=25600, verbose=1, sample_weight=None)

classifier = Model(input=input_img,output=classify_layer)

classifier.compile(optimizer=keras.optimizers.SGD(lr = 0.011),
                    loss='binary_crossentropy',metrics=['accuracy'])

class_wt = {0 : 1.,
    1: 10.}

classifier.fit(x_train,
                y_train,
                batch_size=2500,
                nb_epoch=3,
                callbacks=[tbCallback],
                class_weight = class_wt)

xtp = classifier.predict(x_test)

hidden_layer = Dense(3,activation='sigmoid')()

classify_layer2 = Dense(1, activation='sigmoid')
