import numpy as np
import numpy.ma as ma
import os
import tensorflow as tf


from lib.utils import bn_conv_relu, bn_upconv_relu
from lib.unet_generator import UnetGenerator

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)

from tensorflow.keras.layers import (
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

from tensorflow.keras.models import Model

# ensure reproduceability
np.random.seed(1)


class BaseModel:

    def __init__(self, config):

        self.model = None
        self.config = config
        self.model_save_path = str(self.config["model_path"])
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
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


    def tf_dataset_from_generator(self, generator):
        return tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(
                            None,
                            self.config['input_size'],
                            self.config['input_size'],
                            self.config['total_bands']
                        ),
                        dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(
                            None,
                            self.config['input_size'],
                            self.config['input_size'],
                            1
                        ), dtype=tf.int32
                    )
                )
            ).with_options(self.options)


    def train(self):
        train_generator = UnetGenerator(
            self.config['train_dir'],
            n_channels=self.config['total_bands']
        )
        val_generator = UnetGenerator(
            self.config['val_input_dir'],
            n_channels=self.config['total_bands']
        )
        train_dataset = self.tf_dataset_from_generator(train_generator)
        val_dataset = self.tf_dataset_from_generator(val_generator)

        results = self.model.fit(
            train_dataset,
            epochs=200,
            steps_per_epoch=np.floor(
                train_generator.num_samples / train_generator.batch_size
            ),
            validation_data=val_dataset,
            validation_steps=np.floor(
                val_generator.num_samples / val_generator.batch_size
            ),
            callbacks=self.callbacks
        )

        return results
