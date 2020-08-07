from train import Trainer
import json
import gc

experiment = {'drop_1': [1, 2, 3, 4, 5],
              'drop_2': [0, 2, 3, 4, 5],
              'drop_3': [0, 1, 3, 4, 5],
              'just_1': [0, 3, 4, 5],
              'just_2': [1, 3, 4, 5],
              'just_3': [2, 3, 4, 5],
              'drop_56': [0, 1, 2, 3],
              'baseline': [0, 1, 2, 3, 4, 5]}


def experiment_input_generator(bands, method='just', baseline=[0, 1, 2, 3, 4, 5]):
    """Generates a dictionary containing the experiment name and band list

    Args:
        bands ([ints]): List of integers representing bands to include or exclude
        method (str, optional): Use 'just' to only use bands included in bands, or 'drop' to use
            the baseline with the bands excluded. Defaults to 'just'.
        baseline ([ints], optional): List of integers representing bands in the baseline.
            Defaults to [0,1,2,3,4,5].

    Returns:
        dict: {experiment_name: band_list}
    """

    valid_methods = ['just', 'drop']

    if method not in valid_methods:
        raise ValueError("method arg must be 'just' or 'drop'")

    if not (type(bands) is list and all([type(band) is int for band in bands])):
        raise ValueError("bands must be a list of integers")

    if not (type(baseline) is list and all([type(band) is int for band in baseline])):
        raise ValueError("baseline must be a list of integers")

    if not all([band in baseline for band in bands]):
        raise ValueError("all bands must be contained in baseline")

    name = f'{method}_{''.join(str(band) for band in bands)}'

    if method == 'drop':
        final_bands = list(set(baseline) - set(bands))

    elif method == 'just':
        final_bands = bands

    return {'experiment_name': name, 'band_list': final_bands}


with open("config.json", 'r') as file:
    config = json.load(file)

model_path = config["model_path"].replace('.h5', '')

for title, bands in experiment.items():
    print("experiment, {}:".format(title))
    config["bands"] = bands
    config["model_path"] = f'{model_path}_{title}.h5'
    trainer = Trainer(config)
    trainer.train()
    del trainer
    gc.collect()
