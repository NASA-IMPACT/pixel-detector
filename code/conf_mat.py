import matplotlib
matplotlib.use('Agg')

from data_preparer import PixelDataPreparer
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from config import PREDICT_THRESHOLD
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np
import os
import seaborn as sn
import pandas as pd

nhood_sizes = [1, 3, 5, 7, 9]


def predict(model, data):
    """give out predictions for the given image
    Returns:
        TYPE: Description
    """
    pred_bmp = model.predict(np.array(data), batch_size=50000)
    return pred_bmp


for n in nhood_sizes:
    dataset = PixelDataPreparer("../data/images_val", neighbour_pixels=n)
    dataset.iterate()
    model_path = '../models/smokev3_{}.h5'.format(str(n))
    model = load_model(model_path)
    print(model.summary())
    import pdb;pdb.set_trace()
    input_data = dataset.dataset
    true_labels = dataset.labels
    total_images = len(dataset.img_dims_list)
    last_idx = 0
    conf_mtx = np.zeros((2, 2))
    prediction_list = []
    for i, img_shape in enumerate(dataset.img_dims_list):
        conf_mtx = np.zeros((2, 2))
        img_prediction = predict(
            model,
            input_data[last_idx:last_idx + img_shape[0] * img_shape[1]]
        )
        img_true = true_labels[last_idx:last_idx + img_shape[0] * img_shape[1]]
        conf_mtx += confusion_matrix(img_true,
                                     img_prediction > PREDICT_THRESHOLD,)
        last_idx += img_shape[0] * img_shape[1]

    tp = conf_mtx[1][1]
    tn = conf_mtx[0][0]
    fp = conf_mtx[0][1]
    fn = conf_mtx[1][0]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    metrics = dict()
    metrics['model'] = str(n)
    metrics['tp'] = tp
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['sensitivity'] = sensitivity
    metrics['specificity'] = specificity
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['accuracy'] = accuracy
    metrics['f1_score'] = f1_score

    # df_cm = pd.DataFrame(conf_mtx, range(2),
    #                      range(2))
    # import seaborn as sn
    # sn.set(font_scale=1.4)  # for label size
    # cf = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    # cf.get_figure().savefig('../conf_mat_{}.png'.format(str(n)))
    # cf = None
    # sn = None
    directory = '../{}/'.format(str(n))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('../{}_metrics.txt'.format(str(n)), 'w') as txt:
        txt.write(str(metrics))

