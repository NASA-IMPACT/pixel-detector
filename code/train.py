# -*- coding: utf-8 -*-
# @Author: Muthukumaran R.
# @Date:   2019-04-02 04:31:43
# @Last Modified by:   Muthukumaran R.
# @Last Modified time: 2019-08-23 13:25:21

import json

from models import PixelModel  # , UNetModel


MODELS = {
    'pixel': PixelModel,
    # 'unet': UNetModel,
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
    trainer = Trainer(json.load(open('config.json')))
    trainer.train()
