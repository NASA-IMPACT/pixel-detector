import json

from model import PixelModel, DeconvModel

MODELS = {
    'pixel': PixelModel,
    'conv': DeconvModel
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
