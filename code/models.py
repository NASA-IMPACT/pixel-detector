import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import tensorflow as tf


from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    BatchNormalization,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    UpSampling2D,
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_preparer import PixelDataPreparer
from unet_generator import UnetGenerator
np.random.seed(1)


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

        visible = Input(
            shape=(
                self.num_neighbor * 2,
                self.num_neighbor * 2,
                len(self.bands)
            )
        )

        conv1 = Conv2D(
            32, kernel_size=2, activation="relu",
            padding="same"
        )(visible)

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
            np.array(x),
            np.array(y),
            nb_epoch=self.config["num_epoch"],
            batch_size=self.config["batch_size"],
            steps_per_epoch=5,
            callbacks=self.callbacks,
            validation_split=0.30,
            shuffle=True,
            max_queue_size=5,
            use_multiprocessing=True,
        )

    def save_model(self):

        self.model.save(self.savepath)


class BaseModel:

    def __init__(self, config):

        self.model = None
        self.config = config
        self.model_save_path = str(self.config["model_path"])
        self.create_model()
        self.build_callbacks()
        self.build_model()

    def build_callbacks(self):
        log_path = self.model_save_path
        base_path = os.path.splitext(log_path)[0]
        log_path = base_path + '.log'
        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=20,
                          verbose=1, mode="auto"),
            ModelCheckpoint(filepath=self.model_save_path,
                            verbose=1, save_best_only=True),
            CSVLogger(log_path, append=True),

        ]

    def build_model(self):

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print(self.model.summary())


class UNetModel(BaseModel):

    def create_model(self):
        num_layers = 3
        input_shape = (
            self.config['input_size'],
            self.config['input_size'],
            self.config['total_bands']
        )
        inputs = Input(input_shape)

        filters = 24
        upconv_filters = 32

        kernel_size = (3, 3)
        activation = 'relu'
        strides = (1, 1)
        padding = 'same'
        kernel_initializer = 'he_normal'
        output_activation = 'sigmoid'

        conv2d_args = {
            'kernel_size': kernel_size,
            'activation': activation,
            'strides': strides,
            'padding': padding,
            'kernel_initializer': kernel_initializer
        }

        conv2d_trans_args = {
            'kernel_size': kernel_size,
            'activation': activation,
            'strides': (2, 2),
            'padding': padding,
        }

        bachnorm_momentum = 0.01

        pool_size = (2, 2)
        pool_strides = (2, 2)
        pool_padding = 'valid'

        maxpool2d_args = {
            'pool_size': pool_size,
            'strides': pool_strides,
            'padding': pool_padding,
        }

        x = Conv2D(filters, **conv2d_args)(inputs)
        c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

        down_layers = []

        for l in range(num_layers):
            x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
            x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
            down_layers.append(x)
            x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
            x = MaxPooling2D(**maxpool2d_args)(x)

        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

        for conv in reversed(down_layers):
            x = concatenate([x, conv])
            x = bn_conv_relu(
                x, upconv_filters, bachnorm_momentum, **conv2d_args
            )
            x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
            x = bn_upconv_relu(
                x, filters, bachnorm_momentum, **conv2d_trans_args
            )

        x = concatenate([x, c1])
        x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

        outputs = Conv2D(
            1,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=output_activation,
            padding='valid'
        )(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        self.model = model

    def train(self):

        train_generator = UnetGenerator(
            self.config['train_dir'],
            n_channels=self.config['total_bands']
        )
        val_generator = UnetGenerator(
            self.config['val_input_dir'],
            n_channels=self.config['total_bands']
        )
        results = self.model.fit(
            train_generator,
            epochs=200,
            steps_per_epoch=np.floor(
                train_generator.num_samples / train_generator.batch_size
            ),
            validation_data=val_generator,
            validation_steps=np.floor(
                val_generator.num_samples / val_generator.batch_size
            ),
            callbacks=self.callbacks,
        )
        self.infer(self.model_save_path)
        return results


def infer(model_path, val_input_path, val_output_path, threshold):
    model = load_model(model_path)
    val_generator = UnetGenerator(
        val_input_path, batch_size=1, to_fit=False
    )
    visualize_results(val_generator, model, val_output_path, val_input_path, threshold)

def visualize_results(val_generator, model, save_path, val_input_path, threshold):

    if not os.path.exists:
        os.mkdirs(save_path)
    
    name = sorted([i for i in os.listdir(val_input_path) if i[0] != "."])
    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()
    
    for i, batch_data in enumerate(val_generator):
        input_batch, bmp_batch = batch_data, batch_data
        bmp_predict_batch = model.predict(input_batch)

        for j in range(len(input_batch)):
            
            plt.imshow(convert_rgb(input_batch[j]).astype('uint8'))  
            
            plt.imshow(
                ma.masked_where(bmp_predict_batch[j] < threshold, bmp_predict_batch[j])[:, :, 0],
                alpha=0.45,   
                cmap='spring'
            )  
            
            name = name[1].replace('tif', 'png')
            plt.savefig(os.path.join(save_path, name),  dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
            plt.clf()
            


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x

def convert_rgb(img):

    red = img[:, :, 1].astype('uint8')
    blue = img[:, :, 0].astype('uint8')
    pseudo_green = img[:, :, 2].astype('uint8')
    height, width = red.shape

    img = np.moveaxis(
        np.array([red, pseudo_green, blue]), 0, -1
    )

    return img


def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a, b

    Args:
        a (list/array): data a
        b (list/array): data a

    Returns:
        TYPE: a,b shuffled and resampled
    """

    assert len(a) == len(b)

    indices = np.random.permutation(len(a))

    return [
           [a[index] for index in indices],
           [b[index] for index in indices]
    ]

if __name__ == '__main__':
    infer()
