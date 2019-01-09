import datetime
import glob
import json
import os

from config import (
    IMG_DIR,
    BANDS_LIST,
    SHAPEFILE_DIR
)


def __main__():
    group = 'train_list'
    num_bands = BANDS_LIST.__len__()
    object_dicts = []
    shp_date_format = '%Y%m%d'

    for shp_file in os.listdir(SHAPEFILE_DIR):
        img_list = [''] * num_bands
        if '.shp' in shp_file and '.xml' not in shp_file:  # make sure the shp.xml files are not read

            date_str = shp_file[-12:-4]
            st = datetime.datetime.strptime(date_str, shp_date_format)
            tt = st.timetuple()
            dbf_file = shp_file[:-4] + '.dbf'
            shx_file = shp_file[:-4] + '.shx'
            if os.path.exists(SHAPEFILE_DIR + '/' + dbf_file) and os.path.exists(SHAPEFILE_DIR + '/' + shx_file):
                # if this is a valid shapefile (shp,dbf,shx pair)
                for file in glob.glob(IMG_DIR + '/*' + str(tt.tm_year * 1000 + tt.tm_yday) + '*.tif'):
                    for j in range(num_bands):
                        if '_' + BANDS_LIST[j] + '_' in file:
                            img_list[j] = file
                    if '' not in img_list:
                        object_dict = {
                            "image_paths": img_list,
                            "shapefile_path": SHAPEFILE_DIR + '/' + shp_file
                        }
                        object_dicts.append(object_dict)

    filename = "{}.json".format(group)
    with open(filename, "w") as stream:
        json.dump(object_dicts, stream)


if __name__ == "__main__":
    __main__()
