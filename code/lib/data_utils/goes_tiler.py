from typing import Any, Tuple, Dict, List

import io
import os
from concurrent import futures
import mercantile

import numpy

from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from rio_tiler.reader import part
from rio_tiler.utils import render
from rio_tiler.profiles import img_profiles


def calculate_tile_bounds(epsg_code: int, x: int, y: int, z: int) -> List:
    """
    Calculate the tile bounds given the EPSG code and the x, y, z values
    Args:
        epsg_code: EPSG code
        x: column identifier
        y: column identifier

    """
    if epsg_code == 3857:
        tile_bounds = mercantile.xy_bounds(mercantile.Tile(x=x, y=y, z=z))
    elif epsg_code == 4326:
        tile_bounds = mercantile.bounds(mercantile.Tile(x=x, y=y, z=z))
    else:
        raise Exception(f"epsg code {epsg_code} is not supported")
    return list(tile_bounds)


def goes_tile(
    src_dst,
    tile_x: int,
    tile_y: int,
    tile_z: int,
    tilesize: int = 256,
    unscale: bool = True,
    epsg_code: int = 3857,
    **kwargs: Any,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Create mercator tile from GOES16 data.
    Attributes
    ----------
        src_path : str
            GOES 16 COG path.
        tile_x : int
            Mercator tile X index.
        tile_y : int
            Mercator tile Y index.
        tile_z : int
        kwargs: dict, optional
            These will be passed to the 'rio_tiler.utils._tile_read' function.
    Returns
    -------
        data : numpy ndarray
        mask: numpy array
    """
    bounds = transform_bounds(
        src_dst.crs, "epsg:4326", *src_dst.bounds, densify_pts=21
    )
    mercator_tile = mercantile.Tile(x=tile_x, y=tile_y, z=tile_z)
    dst_crs = CRS.from_epsg(epsg_code)
    tile_bounds = calculate_tile_bounds(epsg_code, x, y, z)

    data, mask = part(
        src_dst, tile_bounds, tilesize=tilesize, dst_crs=dst_crs, **kwargs
    )

    if unscale:
        data = data.astype("float32", casting="unsafe")
        numpy.multiply(data, src_dst.scales[0], out=data, casting="unsafe")
        numpy.add(data, src_dst.offsets[0], out=data, casting="unsafe")

    return data, mask, src_dst.tags()


def _geotiff_options(
    x: int,
    y: int,
    z: int,
    tile_size: int = 256,
    epsg_code: int = 3857
) -> Dict:
    """Return rasterio options for GeoTIFF."""
    tile_bounds = calculate_tile_bounds(epsg_code, x, y, z)

    return dict(
        crs=CRS.from_epsg(epsg_code),
        transform=from_bounds(*tile_bounds, tile_size, tile_size),
    )


def _cog_path(sceneid, band):
    meta = _parse_cog_sceneid(sceneid)
    return os.path.join(f"s3://{cog_bucket}", meta["key"], f"B{band}.tif")


def tiles(
    z: int,
    x: int,
    y: int,
    rio_array: list,
    scale: int = 1,
    ext: str = "tif",
    epsg_code: int = 4326,
    sceneid: str = None,
    rescale: str = None,
    color_ops: str = None,
    color_map: str = None,
    resampling_method: str = "bilinear",
    unscale: bool = True,
):
    """Handle tile requests."""
    if not sceneid:
        return ("NOK", "text/plain", "Missing 'sceneid' parameter")

    if isinstance(scale, str):
        scale = int(scale)

    if isinstance(unscale, str) and unscale.upper() == "FALSE":
        unscale = False

    tilesize = 256 * scale

    def _tile(band):
        src_path = rio_array[band]
        return goes_tile(
            src_path,
            x,
            y,
            z,
            tilesize=tilesize,
            unscale=unscale,
            epsg_code=epsg_code,
            resampling_method=resampling_method,
        )

    with futures.ThreadPoolExecutor() as executor:
        tile, masks, base_options = zip(*list(executor.map(_tile, range(0,6))))
        tile = numpy.concatenate(tile)
        mask = numpy.all(masks, axis=0).astype(numpy.uint8) * 255
    rtile = tile
    driver = "jpeg" if ext == "jpg" else ext
    options = img_profiles.get(driver, {})
    if ext == "tif":
        ext = "tiff"
        driver = "GTiff"
        tile_options: Dict = _geotiff_options(
            x, y, z, tile_size=tilesize, epsg_code=epsg_code
        )
        auxillary_options = dict()
        for i, band_option in enumerate(base_options):
            auxillary_options[f'b{i}_kappa0'] = band_option.get('kappa0')
        auxillary_options['t'] = band_option['t']
        tile_options.update(dict(nodata=-9999.0))
        rtile[:, mask == 0] = -9999.0

    if ext == "npy":
        sio = io.BytesIO()
        numpy.save(sio, (rtile, mask))
        sio.seek(0)
        return ("OK", "application/x-binary", sio.getvalue())
    else:
        return render(
            rtile, mask, img_format=driver, color_map=color_map, **tile_options
        ), auxillary_options
