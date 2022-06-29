import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose
)

from .unet_generator import UnetGenerator

np.random.seed(1)


def bn_conv_relu(_input, filters, batchnorm_momentum, **conv2d_args):
    """
    Method to add a combination of BatchNormalization and Conv2D
    Args:
        _input: input layer to add batch normalization and conv2d to
        filters: Filter details for convolution layer
        batchnorm_momentum: Momentum for batch normalization
        conv2d_args: all remaining arguments needed for convolution layer
    """
    x = BatchNormalization(momentum=batchnorm_momentum)(_input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(_input, filters, batchnorm_momentum, **conv2d_trans_args):
    """
    add a combination of BatchNormalization and 2d Convolution Transpose to passed input layer
    Args:
        _input: input layer to add batch normalization and 2d convolution transpose to
        filters: Filter details for convolution transpose layer
        batchnorm_momentum: Momentum for batch normalization
        conv2d_args: all remaining arguments needed for convolution transpose layer
    """
    x = BatchNormalization(momentum=batchnorm_momentum)(_input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


def convert_rgb(img):
    """
    Convert passed list of bands to proper RGB image for visualization purposes
    Args:
        img: numpy array of GOES data (6 bands)
    """
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


def infer(model_path, val_input_path, val_output_path):
    """
    Load the model and infer on the provided examples, store the output to a directory

    Args:
        model_path (str): Model path
        val_input_path (str): Validation data path
        val_output_path (str): Inference storage path

    Returns:
        None
    """
    model = load_model(model_path)
    val_generator = UnetGenerator(
        val_input_path, batch_size=4, to_fit=False
    )
    visualize_results(val_generator, model, val_output_path)


def visualize_results(val_generator, model, save_path):
    """
    Infer results and save results to a directory

    Args:
        val_generator (UnetGenerator): Data generator instance
        model (keras.Model): Keras model
        save_path (str): Inference storage path

    Returns:
        None
    """
    if not os.path.exists:
        os.mkdirs(save_path)

    f, ax = plt.subplots(1, 2)

    for index, batch_data in enumerate(val_generator):
        input_batch, bmp_batch = batch_data, batch_data
        bmp_predict_batch = model.predict(input_batch)

        for batch_index in range(len(input_batch)):
            ax[0].imshow(
                convert_rgb(input_batch[batch_index]).astype('uint8')
            )
            ax[1].imshow(convert_rgb(input_batch[batch_index]).astype('uint8'))
            bmp_data = bmp_batch[batch_index].astype('uint8')
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

            plt.savefig(os.path.join(save_path, f'{i}_{j}.png'))
