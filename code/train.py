import json

from models import PixelModel, UNetModel


MODELS = {
    'pixel': PixelModel,
    'unet': UNetModel,
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
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        # model building
        tf.keras.backend.clear_session()  # For easy reset of notebook state.

        print('Trying to resolve cluster')
        cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=1)
        slurm_resolver = SlurmClusterResolver()
        print(slurm_resolver.cluster_spec())
        print('Resolved cluster')
        mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)
        print('Number of replicas:', mirrored_strategy.num_replicas_in_sync)
        with mirrored_strategy.scope():
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
