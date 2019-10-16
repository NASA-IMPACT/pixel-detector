import numpy as np

from data_preparer import PixelDataPreparer
from sklearn import svm


class SVMModel():

    def __init__(self, config):

        self.config = config
        self.num_neighbor = self.config["num_neighbor"]
        self.savepath = str(self.config["model_path"])
        self.make_model()
        self.train()

    def make_model(self):
        """
            Make the model
        """

        self.model = svm.SVC(gamma='scale')

    def train(self):

        dp = PixelDataPreparer(
            '../data/images_train_no_cza/', neighbour_pixels=self.num_neighbor
        )
        dp.iterate()
        x = np.array(dp.dataset)
        y = np.array(dp.labels)
        x, y = unison_shuffled_copies(x, y)

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "mae"]
        )

        self.model.fit(
            x,
            y,
        )
        print(self.model.coef_)

    def save_model(self):

        self.model.save(self.savepath)


def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a,b

    Args:
        a (list/array): data a
        b (list/array): data a

    Returns:
        TYPE: a,b shuffled and resampled
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
