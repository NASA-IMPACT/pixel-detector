import json
import tensorflow as tf

from lib.slurm_utils.slurm_cluster_resolver import SlurmClusterResolver

from models import UNetModel
from tensorflow import keras
from tensorflow.keras import layers


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
        # model building
        tf.keras.backend.clear_session()  # For easy reset of notebook state.

        slurm_resolver = SlurmClusterResolver()
        mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(
                cluster_resolver=slurm_resolver
            )
        print('Number of replicas:', mirrored_strategy.num_replicas_in_sync)

        with mirrored_strategy.scope():
            self.model = UNetModel(self.config)


    def train(self):
        """
        Alias to model holder train method
        """
        self.model.train()


    def load(self):
        """
        Alias to model holder load method
        """
        self.model.load()


if __name__ == '__main__':
    trainer = Trainer(json.load(open('code/config.json'))) 
    trainer.train()

