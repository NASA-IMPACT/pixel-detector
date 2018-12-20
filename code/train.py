import numpy as np
import tensorflow
import keras
import json
from model import PixelModel
import argparse

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

argparser.add_argument(
    '-e',
    '--epoch',
    help='number of epoch')

MODELS = {
            'pixel':PixelModel,
            'conv':ConvModel
         }

class Trainer:

    def __init__(self, config):
        """
        config = {
            'type': < choice of 'pixel' or 'deconv'>,
            'num_neighbor': < TODO: explain this parameter >,
            ''
        }
        """

        self.config = config
        self.model_holder = MODELS[self.config['type']](self.config)

    def train(self):
        """
        Alias to model holder train method
        """
        self.model_holder.train()

    def load(self):
        """
        Alias to model holder load method
        """
        self.model_holder.load()



if __name__ == '__main__':

    trainer = Trainer({})
    trainer.train()




