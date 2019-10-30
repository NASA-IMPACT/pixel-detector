from train import Trainer
import json

experiment = {'drop_1': [1, 2, 3, 4, 5],
              'drop_2': [0, 2, 3, 4, 5],
              'drop_3': [0, 1, 3, 4, 5],
              'just_1': [0, 3, 4, 5],
              'just_2': [1, 3, 4, 5],
              'just_3': [2, 3, 4, 5],
              'drop_56': [0, 1, 2, 3],
              'baseline': [0, 1, 2, 3, 4, 5]}

with open("config.json", 'r') as file:
    config = json.load(file)

model_path = config["model_path"].replace('.h5', '')

for title, bands in experiment:
    config["bands"] = bands
    config["model_path"] = model_path+"_"+title+".h5"
    Trainer(config).train()
