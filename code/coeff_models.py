import numpy as np

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from data_preparer import PixelDataPreparer
from sklearn.ensemble import RandomForestClassifier


class SVMModel():

    def __init__(self, config):

        self.config = config
        self.num_neighbor = self.config["num_neighbor"]
        self.savepath = str(self.config["model_path"])
        self.make_model()

    def make_model(self):
        """
            Make the model
        """

        # self.model = LinearSVC(
        #     verbose=True,
        #     tol=1e-5,
        # )

        # self.model = RandomForestClassifier(
        #     n_estimators=100,
        #     max_depth=7,
        #     random_state=0,
        #     verbose=True,
        #     n_jobs=5,
        # )
        self.model = PCA()

    def train(self):

        dp = PixelPreparer(
            '../data/images_val_cza/', neighbour_pixels=self.num_neighbor
        )
        dp.iterate()
        x = np.array(dp.dataset)
        nsamples, nx, ny, nbands = x.shape
        x = x.reshape(nsamples, nx * ny * nbands)
        print(x.shape)
        y = np.array(dp.labels)
        print(y.shape)
        x, y = unison_shuffled_copies(x, y)

        self.model.fit(
            x,
        )
        # print(self.aggregate_coeffs(nx, ny, nbands))

    def aggregate_coeffs(self, nx, ny, nbands):
        coef = self.model.feature_importances_
        band_size = nx * ny
        coef_mean = np.zeros((nbands,))
        for band_num in range(nbands):
            coef_mean[band_num] = np.sum(np.abs(
                coef[band_num * band_size: (band_num + 1) * band_size]
            ))
            print(coef_mean[band_num])

        return coef_mean

    def save_model(self):

        self.model.save(self.savepath)


def unison_shuffled_copies(a, b):
    """
    shuffle a,b in unison and return shuffled a,b

    Args:
        a (list/array): data a
        b (list/array): data b

    Returns:
        TYPE: a,b shuffled and resampled
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
