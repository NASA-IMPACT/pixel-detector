import numpy as np

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
)
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
)
from keras.models import Model
from loss_plot import TrainingPlot
from data_preparer import PixelDataPreparer


class PixelModel():

    def __init__(self, config):

        self.config = config
        self.bands = self.config["bands"]
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
                               len(self.bands)
                               )
                        )
        conv1 = Conv2D(32, kernel_size=2, activation="relu",
                       padding="same")(visible)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)
        conv2 = Conv2D(64, kernel_size=2, activation="relu",
                       padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)
        flatten = Flatten()(pool2)
        dense1 = Dense(40, activation="relu")(flatten)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(25, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(10, activation="relu")(dense1)
        dense1 = Dropout(0.3)(dense1)
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
            CSVLogger(self.savepath.replace('h5', 'log'), append=True),
        ]

    def train(self):

        dp = PixelDataPreparer(
            path=self.config["train_img_path"],
            neighbour_pixels=self.num_neighbor
        )
        dp.iterate(self.bands)
        x = np.array(dp.dataset)
        print('input shape', x.shape)
        y = np.array(dp.labels)
        print('label shape', y.shape)
        x, y = unison_shuffled_copies(x, y)

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
    return a[p], b[p]
