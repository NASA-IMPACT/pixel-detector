import numpy as np
import tensorflow
import keras
import json
from model import pixelModel, deconvolution

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

argparser = argparse.ArgumentParser(
    description='train model on given json model')


argparser.add_argument(
    '-m',
    '--model',
    help='type of model')

argparser.add_argument(
    '-j',
    '--jsonfile',
    help='json file with ncfile and shapefile mapping')

argparser.add_argument(
    '-s',
    '--savepath',
    help='path to save model')

argparser.add_argument(
    '-i',
    '--initial',
    help='path to initial model')


def main(args):
    # build model
    if args.model == 'pixel' or args.model == 'p':
        model = pixelModel(num_neighbor = 5)
    
    if args.initial != []:
        model.train(jsonfile,num_epochs = int(args.epoch)
        model.save_model(args.savepath)

    else:
        try:
            model.load(args.initial)
        except:
            print('check path and/or model')



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

