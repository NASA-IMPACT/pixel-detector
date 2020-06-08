import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
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
            np.array(x),
            np.array(y),
            nb_epoch=self.config["num_epoch"],
            batch_size=self.config["batch_size"],
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
        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=20,
                          verbose=1, mode="auto"),
            ModelCheckpoint(filepath=self.model_save_path,
                            verbose=1, save_best_only=True),
            CSVLogger(self.model_save_path.replace('h5', 'log'), append=True),

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
        num_layers = 1
        input_shape = (self.config['input_size'], self.config['input_size'], 6)
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
            padding='valid')(x)

        model = Model(inputs=[inputs], outputs=[outputs])
        self.model = model

    def train(self):
        # data_gen_args = dict(featurewise_center=True,
        #                      featurewise_std_normalization=True,
        #                      rotation_range=90.,
        #                      width_shift_range=0.1,
        #                      height_shift_range=0.1,
        #                      zoom_range=0.2)

        # train_datagen = ImageDataGenerator(**data_gen_args)

        # val_datagen = ImageDataGenerator(rescale=1. / 255)

        # train_image_generator = train_datagen.flow_from_directory(
        #     '../unet_master/train/frames/',
        #     class_mode=None,
        #     target_size=(self.config['input_size'], self.config['input_size']),
        #     batch_size=8,
        #     seed=SEED
        # )

        # train_mask_generator = train_datagen.flow_from_directory(
        #     '../unet_master/train/masks/',
        #     class_mode=None,
        #     target_size=(self.config['input_size'], self.config['input_size']),
        #     batch_size=8,
        #     color_mode='grayscale',
        #     seed=SEED
        # )

        # val_image_generator = val_datagen.flow_from_directory(
        #     '../unet_master/val/frames/',
        #     class_mode=None,
        #     target_size=(self.config['input_size'], self.config['input_size']),
        #     batch_size=4,
        #     seed=SEED,
        # )

        # val_mask_generator = val_datagen.flow_from_directory(
        #     '../unet_master/val/masks/',
        #     class_mode=None,
        #     target_size=(self.config['input_size'], self.config['input_size']),
        #     batch_size=4,
        #     color_mode='grayscale',
        #     seed=SEED,
        # )

        # train_generator = zip(train_image_generator, train_mask_generator)
        # val_generator = zip(val_image_generator, val_mask_generator)
        # # vis_res(val_generator, 1, 2)

        train_generator = UnetGenerator('../unet_master/train/frames/data/')
        val_generator = UnetGenerator('../unet_master/val/frames/data/')
        results = self.model.fit_generator(
            train_generator,
            epochs=200,
            steps_per_epoch=64,
            validation_data=val_generator,
            callbacks=self.callbacks,
            validation_steps=24,
        )
        return results


def infer(model_path):
    model = load_model(model_path)
    val_generator = UnetGenerator('../unet_master/val/frames/data/', batch_size=4)
    vis_res(val_generator, model)


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


def vis_res(val_generator, model):
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    f, ax = plt.subplots(1, 2)
    for i, batch_data in enumerate(val_generator):
        (modis_batch, bmp_batch) = batch_data
        bmp_predict_batch = model.predict(modis_batch)
        for j in range(len(modis_batch)):
            ax[0].imshow(
                convert_rgb(modis_batch[j]).astype('uint8')
            )
            ax[1].imshow(convert_rgb(modis_batch[j]).astype('uint8'))
            bmp_data = bmp_batch[j].astype('uint8')
            ax[0].imshow(
                ma.masked_where(
                    bmp_data != 1, bmp_data
                )[:, :, 0],
                alpha=0.35,
                cmap='Purples'
            )

            ax[1].imshow(
                ma.masked_where(
                    bmp_predict_batch[j] < 0.5, bmp_predict_batch[j]
                )[:, :, 0],
                alpha=0.45,
                cmap='spring'
            )
            plt.savefig(f'../unet_master/results/{i}_{j}.png')


def convert_rgb(img):

    red = img[:, :, 1]
    blue = img[:, :, 0]
    pseudo_green = img[:, :, 2]
    height, width = red.shape
    img = np.moveaxis(
        np.array([red, pseudo_green, blue]), 0, -1
    )

    return img


def vis_res_predict(val_generator, i, j, model):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2)
    import numpy.ma as ma
    for h in range(j):
        (modis_data, bmp_data) = next(val_generator)
    ax[0].imshow(modis_data[i])
    ax[1].imshow(modis_data[i])
    bmp_data = model.predict(modis_data)
    ax[1].imshow(ma.masked_where(bmp_data < 0.1, bmp_data)[i, :,:,0],alpha=0.25,cmap='winter')
    ax[0].imshow(bmp_data[i, :,:,0], alpha=0.25,cmap='winter')
    plt.show()


if __name__ == '__main__':

    infer('../models/smoke_unet.h5')


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
    indices = np.random.permutation(len(a))
    return [
           [a[index] for index in indices],
           [b[index] for index in indices]
    ]
