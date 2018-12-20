
from config import LANDSAT1,EGYPT_SHAPEFILE,NETHERLANDS_SHAPEFILE,OCEAN_SHAPEFILE,WGS84_DIR
from extract_features import extract_features
import numpy as np
from io_util import load_input,load_output
from sklearn.preprocessing import normalize


img_features1,bitmap_tiles1,img_shape = extract_features(LANDSAT3,64,NETHERLANDS_SHAPEFILE)

#img_features2,bitmap_tiles2,img_shape2 = extract_features(LANDSAT2,64,[EGYPT_SHAPEFILE])



inp = load_input(img_features1)

outp = load_output(bitmap_tiles1)


inp_predict = autoencoder.predict(inp.T,batch_size=25000)

img_tiles = get_tiles_from_prediction(inp_predict,bitmap_tiles1)

img = image_from_tiles(img_tiles,64,img_shape)


def get_tiles_from_prediction(predict_matrix, bitmap_tiles1):
    prediction_tiles = bitmap_tiles1
    i = 0
    for tile,(row,col), _ in prediction_tiles:
        tile = tile.astype(np.float32, copy=False)
        tile =  np.reshape(predict_matrix[i*64*64:(i+1)*64*64]*255.0,(64,64))
        prediction_tiles[i][0][:] = tile
        i = i + 1

    return prediction_tiles


def image_from_tiles(tiles, tile_size, image_shape):
    """'Stitch' several tiles back together to form one image."""

    image = np.zeros(image_shape, dtype=np.uint8)

    for tile, (row, col), _ in tiles:
        tile = np.reshape(tile, (tile_size, tile_size))
        image[row:row + tile_size, col:col + tile_size] = tile

    return image
