import numpy as np
import keras.optimizers
import os

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    concatenate,
)

from keras.models import Model
from keras import optimizers
from config import BANDS_LIST
from loss_plot import TrainingPlot
from data_preparer import PixelDataPreparer


class PixelModel():

    def __init__(self, config):

        self.config = config
        self.num_neighbor = self.config["num_neighbor"]
        self.savepath = str(self.config["model_path"])
        self.make_model()
        self.build_callbacks()

    def make_model(self):
        """
            Make the model
        """

        visible = Input(shape=(self.num_neighbor * 2,
                               self.num_neighbor * 2,
                               len(BANDS_LIST)
                               )
                        )

        # conv model
        conv1 = Conv2D(32, kernel_size=2, activation="relu",
                       padding="same")(visible)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)
        conv2 = Conv2D(64, kernel_size=2, activation="relu",
                       padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)
        # conv3 = Conv2D(40, kernel_size=2, activation="relu",
        #                padding="same")(pool2)
        # pool3 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv3)
        flatten = Flatten()(pool2)
        dense1 = Dense(40, activation="relu")(flatten)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(25, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(10, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)

        # dense model
        # dense1 = Flatten()(visible)
        # dense1 = Dense(86, activation="relu")(dense1)
        # dense1 = Dropout(0.3)(dense1)
        # dense1 = Dense(70, activation="relu")(dense1)
        # dense1 = Dropout(0.3)(dense1)
        # dense1 = Dense(64, activation="hard_sigmoid")(dense1)
        # dense1 = Dropout(0.3)(dense1)
        # dense1 = Dense(32, activation="relu")(dense1)
        # dense1 = Dense(16, activation="elu")(dense1)
        # dense1 = Dropout(0.2)(dense1)

        dense2 = Dense(5, activation="relu")(dense1)
        output = Dense(1, activation="sigmoid")(dense2)

        self.model = Model(inputs=visible, outputs=output)

    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)
        except IOError:
            print("the model does not conform with the weights given")

    def build_callbacks(self):
        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=20,
                          verbose=1, mode="auto"),
            ModelCheckpoint(filepath=self.savepath,
                            verbose=1, save_best_only=True),
            TrainingPlot(),
        ]

    def train(self):

        dp = PixelDataPreparer(
            '../data/images_train_no_cza/', neighbour_pixels=self.num_neighbor
        )
        dp.iterate()
        x = np.array(dp.dataset)
        print('input shape', x.shape)
        y = np.array(dp.labels)
        print('label shape', y.shape)
        x, y = unison_shuffled_copies(x, y)

        # x_train = np.concatenate(tuple(x_train), axis=0)
        # y_train = np.concatenate(tuple(y_train), axis=0)
        # x_val = np.concatenate(tuple(x_val), axis=0)
        # y_val = np.concatenate(tuple(y_val), axis=0)

        # x_train, y_train = unison_shuffled_copies(x_train, y_train)
        # x_val, y_val = unison_shuffled_copies(x_val, y_val)

        # x_train = np.concatenate(tuple(x_train), axis=0)
        # y_train = np.concatenate(tuple(y_train), axis=0)
        # x_val = np.concatenate(tuple(x_val), axis=0)
        # y_val = np.concatenate(tuple(y_val), axis=0)

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "mae"]
        )
        print(self.model.summary())

        self.model.fit(
            x,
            y,
            nb_epoch=self.config["num_epoch"],
            batch_size=self.config["batch_size"],
            callbacks=self.callbacks,
            validation_split=0.30,
            shuffle=True,
        )

    def save_model(self):

        self.model.save(self.savepath)


class UNetModel():
    def __init__(self, config):

        self.config = config
        self.num_neighbor = self.config["num_neighbor"]
        self.savepath = str(self.config["model_path"])
        self.make_model()
        self.build_callbacks()

    def make_model(self, input_size=(256, 256, 3)):

        inputs = Input(input_size)

        conv1 = Conv2D(20, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool1)

        # conv1 = Conv2D(64, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(inputs)
        # conv1 = Conv2D(64, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # conv2 = Conv2D(128, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(pool1)
        # conv2 = Conv2D(128, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # conv3 = Conv2D(256, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(pool2)
        # conv3 = Conv2D(256, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # conv4 = Conv2D(512, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(pool3)
        # conv4 = Conv2D(512, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.5)(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # conv5 = Conv2D(1024, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(pool4)
        # conv5 = Conv2D(1024, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv5)
        # drop5 = Dropout(0.5)(conv5)

        # up6 = Conv2D(512, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        # merge6 = concatenate([drop4, up6], axis=3)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(merge6)
        # conv6 = Conv2D(512, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv6)

        # up7 = Conv2D(256, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        # merge7 = concatenate([conv3, up7], axis=3)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(merge7)
        # conv7 = Conv2D(256, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv7)

        # up8 = Conv2D(128, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        # merge8 = concatenate([conv2, up8], axis=3)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(merge8)
        # conv8 = Conv2D(128, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv8)

        # up9 = Conv2D(64, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        # merge9 = concatenate([conv1, up9], axis=3)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(merge9)
        # conv9 = Conv2D(64, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same',
        #                kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv2)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)
        except IOError:
            print("the model does not conform with the weights given")

    def build_callbacks(self):

        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=10,
                          verbose=1, mode="auto"),
            ModelCheckpoint(filepath=self.savepath,
                            verbose=1, save_best_only=True),
            TensorBoard(log_dir="./graph", histogram_freq=0,
                        write_graph=True, write_images=True)
        ]

    def train(self):
        # dp = UnetDataPreparer(jsonfile='', save_path='../data/unet_images')
        # x, y = dp.get_unet_data()
        (x_, y_, _, _) = get_data_unet(
            self.config["jsonfile"], 256)
        num_val_imgs = 28
        x_train = x_[num_val_imgs:]
        y_train = y_[num_val_imgs:]
        x_val = x_[:num_val_imgs]
        y_val = y_[:num_val_imgs]

        print(len(x_))

        x_train = np.concatenate(tuple(x_train), axis=0)
        y_train = np.concatenate(tuple(y_train), axis=0)
        x_val = np.concatenate(tuple(x_val), axis=0)
        y_val = np.concatenate(tuple(y_val), axis=0)

        x_train, y_train = unison_shuffled_copies(x_train, y_train)
        x_val, y_val = unison_shuffled_copies(x_val, y_val)
        x, y = unison_shuffled_copies(x, y)
        # y = y / 255
        # print(y[0])
        # y = np.expand_dims(y, axis=-1)

        # self.model.fit(
        #     x,
        #     y,
        #     nb_epoch=self.config["num_epoch"],
        #     batch_size=self.config["batch_size"],
        #     callbacks=self.callbacks,
        #     validation_split=0.25,
        #     shuffle=True
        # )

    def save_model(self):

        self.model.save(self.savepath)


def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a,b

    Args:
        a (list/array): data a
        b (list/array): data a

    Returns:
        TYPE: a,b shuffled and resampled
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    # return balanced_subsample(a[p], b[p])
    return a[p], b[p]


def histogram_equalize(img):
    # return img
    return cv2.equalizeHist(img.astype('uint8'))
