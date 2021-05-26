import os
import shutil
import math
import random

from glob import glob
from io import BytesIO


DATA_FOLDER = '../data/wmts'


def mkdir(foldername):
    """
    creates folders if 'foldername' doesn't exist
    """
    if os.path.exists(foldername):
        print(f"'{foldername}' folder already exists.")
        return
    os.makedirs(foldername)
    print(f"Created folder: {foldername}")


def delete_folder(foldername):
    """deletes folder and its contents """
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    else:
        print(f"Folder {foldername} doesn't exist.")

def create_split(split, files):
    """
    Clear and create folder with new files.
    split: choice of "train", "test", and "val"
    files: list of tiff file paths

    """
    print(f'Preparing {split} split with {len(files)} examples.')
    folder_name = f"{DATA_FOLDER}/{split}"
    if os.path.exists(folder_name):
        delete_folder(folder_name)
    mkdir(folder_name)
    for filename in files:
        internal_filename = filename.split('/')[-1]
        bitmap_filename = filename.replace('.tif', '.bmp')
        shutil.copyfile(filename, f"{folder_name}/{internal_filename}")
        shutil.copyfile(bitmap_filename, f"{folder_name}/{bitmap_filename.split('/')[-1]}")

# Prepare train, val, and test splits
def prepare_splits(source_folder, splits={'train': 0.7, 'val': 0.3}):
    """ Creates training, validation and test splits from `source folder`
    """
    files = glob(f"{source_folder}/*.tif")
    print(f"Total examples found: {len(files)}")
    random.shuffle(files)
    length = len(files)
    train_limit = math.ceil(length * splits['train'])
    # val_limit = train_limit + math.ceil(length * splits['train'])
    create_split('train', files[0:train_limit])
    create_split('val', files[train_limit:])


prepare_splits('../data/wmts_processed_train_val')
