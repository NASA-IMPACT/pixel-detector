import matplotlib  # should be declared before declaring any other package that uses MPL
matplotlib.use("Agg")

import numpy as np
import random

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
    Dropout
)
from keras.models import Model
from preprocessing import get_arrays_from_json, unison_shuffled_copies
from operator import itemgetter


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

        visible = Input(shape=(self.num_neighbor, self.num_neighbor, 8,))

        # conv model
        # conv1   = Conv2D(4, kernel_size=2, activation="relu", padding="same")(visible)
        # pool1   = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)
        # conv2   = Conv2D(2, kernel_size=2, activation="relu", padding="same")(pool1)
        # flatten = Flatten()(conv2)
        # #flatten = Flatten()(visible)
        # dense1  = Dense(32, activation="relu")(flatten)
        # dense2  = Dense(16, activation="relu")(dense1)
        # dense2  = Dropout(0.5)(dense2)

        # dense model
        dense1 = Flatten()(visible)
        dense1 = Dense(100, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(75, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(50, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(25, activation="relu")(dense1)
        dense1 = Dropout(0.2)(dense1)
        dense1 = Dense(15, activation="relu")(dense1)
        dense1 = Dropout(0.1)(dense1)
        dense1 = Dense(10, activation="relu")(dense1)

        dense2 = Dense(5, activation="relu")(dense1)
        output = Dense(1, activation="sigmoid")(dense2)

        self.model = Model(inputs=visible, outputs=output)
        print(self.model.summary())

    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)
        except (InputError):
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

        x_, y_, _ = get_arrays_from_json(
            self.config["jsonfile"], self.num_neighbor)
        num_val_imgs = random.sample(range(0, len(x_)), 5)
        num_val_imgs = 18
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

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        history = self.model.fit(
            x_train,
            y_train,
            nb_epoch=self.config["num_epoch"],
            batch_size=self.config["batch_size"],
            callbacks=self.callbacks,
            validation_data=[x_val, y_val],
            shuffle=True
        )

    def save_model(self):

        self.model.save(self.savepath)


class DeconvModel():
    def __init__(self, num_neighbor=5):

        self.num_neighbor = num_neighbor

        self.make_model()

    def make_model(self):

        # Make the model

        visible = Input(shape=(self.num_neighbor, self.num_neighbor, 8,))
        x = ZeroPadding2D(((2, 1), (1, 2)))(visible)
        x = Conv2D(10, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((3, 3), padding="same")(x)
        x = Conv2D(15, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(10, (2, 2), activation="relu", padding="same")(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(5, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((3, 3))(x)
        x = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
        x = Cropping2D(((2, 2), (2, 2)))(x)

        self.model = Model(visible, x)
        self.model.summary()

    def load_weights(self, weight_path):
        try:
            self.model.load_weights(weight_path)
        except:
            print("the model does not conform with the weights given")

    def train(self, jsonfile, num_epoch, savepath):

        x_train, y_train = get_arrays_from_json(jsonfile, self.num_neighbor)

        self.model.compile(optimizer="adam",
                           loss="mse",
                           metrics=["mae"])

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, mode="auto")
        checkpointer = ModelCheckpoint(
            filepath=savepath, verbose=1, save_best_only=True)

        self.model.fit(
            x=X_train,
            y=np.expand_dims(Y_train, axis=-1),
            batch_size=32, epochs=200, verbose=1, validation_split=0.2,
            callbacks=[early_stopping, checkpointer],
            shuffle=True
        )

    def save_model(self, savepath):
        self.model.save(self.savepath)
