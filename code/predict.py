import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from config import BANDS_LIST
from keras.models import load_model
from PIL import Image
from preprocessing import (
    convert_pixels_to_groups,
)
from tif_utils import (
    create_array_from_nc,
    band_list
)

matplotlib.use('Agg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Predict model on given GOES 16 NC files and model')


argparser.add_argument(
    '-m',
    '--model',
    help='path to trained model')

argparser.add_argument(
    '-i',
    '--input',
    help='path to NC files')

argparser.add_argument(
    '-ns',
    '--neighborhoodSize',
    help='size of neighborhood to take. (must match the model input)')

argparser.add_argument(
    '-f',
    '--folder',
    help='folder to test images')

argparser.add_argument(
    '-v',
    '--visualize',
    help='bitmap to visualize output')


def _main_(args):
    print(args)
    model = load_model(args.model)
    n_size = args.neighborhoodSize
    ncFiles_path = args.input
    X_mtx, _ = create_array_from_nc(band_list(ncFiles_path, BANDS_LIST, time=''), fname='')
    model.summary()
    print('predicting...')
    y_out = model.predict(
        convert_pixels_to_groups(X_mtx),
        batch_size=10000)
    y_mat = np.asarray(y_out * 255.0, dtype='uint8').reshape(X_mtx[:, :, 0].shape)
    Image.fromarray(y_mat).save('test.bmp')
    if not args.visualize == '':

        fig = plt.figure(figsize=(12, 10), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(X_mtx[:, :, 0])
        ax.imshow(y_mat, alpha=0.5, cmap='gray')
        # temp_y_out = y_out
        # ax.imshow(temp_y_out, alpha=0.3, cmap='jet')
        plt.savefig('test.png')
        plt.close()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
