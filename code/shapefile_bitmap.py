import fiona
import io_util as io
import numpy as np
import os
import rasterio
import rasterio.features
import rasterio.warp

from config import BITMAPS_DIR


def create_bitmap(raster_dataset, shapefile_path, satellite_path):
    # type: (object, object, object) -> object
    """Create the bitmap for a given satellite image."""

    satellite_img_name = io.get_file_name(satellite_path)
    cache_file_name = "{}_smoke.tif".format(satellite_img_name)
    cache_path = os.path.join(BITMAPS_DIR, cache_file_name)

    water_features = np.empty((0, ))

    print("Create bitmap for water features.")

    # for shapefile_path in shapefile_paths:

    try:
        print("Load shapefile {}.".format(shapefile_path))
        with fiona.open(shapefile_path) as shapefile:
            # Each feature in the shapefile also contains meta information such as
            # wether the features is a lake or a river. We only care about the geometry
            # of the feature i.e. where it is located and what shape it has.
            geometries = [feature['geometry'] for feature in shapefile]

            # TODO: properties unused.
            properties = [feature['properties'] for feature in shapefile]

            water_features = np.concatenate(
                (water_features, geometries), axis=0)
    except IOError:
        print("No shapefile found.")
        return 0

# Now that we have the vector data of all water features in our satellite image
# we "burn it" into a new raster so that we get a B/W image with water features
# in white and the rest in black. We choose the value 255 so that there is a stark
# contrast between water and non-water pixels. This is only for visualisation
# purposes. For the classifier we use 0s and 1s.
    bitmap_image = rasterio.features.rasterize(
        ((g, 255) for g in water_features),
        out_shape=raster_dataset.shape,
        transform=raster_dataset.transform)

    io.save_bitmap(cache_path, bitmap_image, raster_dataset)

    bitmap_image[bitmap_image == 255] = 1
    return bitmap_image


def remove_edge_tiles(tiled_bands, tiled_bitmap, tile_size, source_shape):
    """Remove tiles which are on the edge of the satellite image and which contain blacked out
    content."""

    EDGE_BUFFER = 350

    rows, cols = source_shape[0], source_shape[1]

    bands = []
    bitmap = []
    for i, (tile, (row, col), _) in enumerate(tiled_bands):
        is_in_center = EDGE_BUFFER <= row and row <= (
            rows - EDGE_BUFFER) and EDGE_BUFFER <= col and col <= (
                cols - EDGE_BUFFER)

        contains_black_pixel = [0, 0, 0] in tile
        is_edge_tile = contains_black_pixel and not is_in_center
        if not is_edge_tile:
            bands.append(tiled_bands[i])
            bitmap.append(tiled_bitmap[i])

    return bands, bitmap, [x[1] for x in bands]
