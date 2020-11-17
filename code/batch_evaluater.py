import matplotlib; matplotlib.use('Agg')

from glob import glob
from evaluate import Evaluate
from keras.models import load_model
from data_preparer import PixelListPreparer


class BatchEvaluate(Evaluate):

    def __init__(self, config, step_num=3, batch_size=50000):
        """init for Evaluate class
        """

        self.batch_size = config['batch_size']
        self.num_n = config['num_neighbor']
        self.model_path = config['model_path']
        self.val_dir = config['val_dir']
        self.save_dir = config['val_output_dir']
        self.model = load_model(self.model_path)

        print(self.model.summary())

        path_list = glob(self.val_dir + '/*.tif')
        paths_lists = [
            path_list[x:x + step_num] for x in range( 0, len(path_list), step_num)
        ]

        for paths in paths_lists:
            self.dataset = PixelListPreparer(
                paths,
                neighbour_pixels=self.num_n
            )
            self.dataset.iterate()
            self.evaluate()
