"""translator."""
import datetime
import os
from typing import Dict
import numpy
import rasterio
from affine import Affine
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from rio_cogeo.profiles import cog_profiles
from rasterio.rio.overview import get_maximum_overview_level

# From https://github.com/Solcast/netcdf-tiff
def get_goes_transform(resolution: str) -> Affine:
    """Return GOES16 geotransform for corresponding resolution."""
    if resolution.startswith("0.5km"):
        return Affine.from_gdal(
            -5434895.081637931,
            501.0043288718853,
            0,
            -5434894.837566491,
            0,
            501.0043288718853,
        )
    elif resolution.startswith("1km"):
        return Affine.from_gdal(
            -5434894.954752678,
            1002.0086577437705,
            0,
            -5434894.964451744,
            0,
            1002.0086577437705,
        )
    elif resolution.startswith("2km"):
        return Affine.from_gdal(
            -5434894.700982173,
            2004.0173154875410,
            0,
            -5434895.218222249,
            0,
            2004.0173154875410,
        )
    else:
        raise Exception(f"Invalid resolution: {resolution}")

def create_cogeo(src_path: str, out_path: str):
    """Convert a GOES netcdf to COG."""
    dataset = "Rad",
    profile  = "deflate",
    profile_options = {"blockxsize": 128, "blockysize": 128},
    ds = Dataset(src_path, "r")
    # Get Projection Info from NetCDF Variables
    proj_name = ds.variables[dataset].getncattr("grid_mapping")
    proj_info = ds.variables[proj_name]
    # List of variable to add to the output COG
    kappa0: float = ds.variables["kappa0"].getValue().tolist()
    t: float = (
        ds.variables["t"].getValue().tolist()
        + datetime.datetime(2000, 1, 1, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
    )
    esun: float = ds.variables["esun"].getValue().tolist()
    attrs: Dict = {
        k: ds.variables["Rad"].getncattr(k) for k in ds.variables["Rad"].ncattrs()
    }
    attrs.update(dict(NETCDF_VARNAME="Rad"))
    # Create PROJ4 string for the input projection (geos)
    # From https://github.com/Solcast/netcdf-tiff
    proj_string = " ".join(
        [
            "+proj=geos",  # The projection name is very important this means geostationary
            "+lon_0={0}".format(proj_info.longitude_of_projection_origin),
            "+h={0}".format(proj_info.perspective_point_height),
            "+a={0}".format(proj_info.semi_major_axis),
            "+b={0}".format(proj_info.semi_minor_axis),
            "+units={0}".format("m"),
            "+sweep={0}".format(proj_info.sweep_angle_axis),
            "+no_defs",
        ]
    )
    # Create Rasterio CRS object from the proj4 string
    crs = CRS.from_string(proj_string)
    # Create GDAL transform
    transform = get_goes_transform(ds.getncattr("spatial_resolution"))
    ds.set_auto_scale(False)
    # Extract Data (pixel) from the netcdf file to an array
    extracted_data = numpy.flip(ds.variables[dataset][:], axis=0)
    height, width = extracted_data.shape
    del ds
    config = dict(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_OVR_BLOCKSIZE="128")
    cogeo_profile = cog_profiles.get(profile)
    cogeo_profile.update(dict(BIGTIFF="IF_SAFER"))
    cogeo_profile.update(profile_options)
    tilesize = min(int(cogeo_profile["blockxsize"]), int(cogeo_profile["blockysize"]))
    with rasterio.Env(**config):
        src_bounds = array_bounds(height, width, transform)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, "epsg:4326", width, height, *src_bounds
        )
        output_profile = dict(
            driver="GTiff",
            dtype=numpy.int16,
            count=1,
            height=dst_height,
            width=dst_width,
            crs="epsg:4326",
            transform=dst_transform,
            nodata=attrs["_FillValue"],
            tiled=True,
            compress="deflate",
            blockxsize=tilesize,
            blockysize=tilesize,
        )
        # Reproject the input file to WGS84 into a MemoryFile
        with MemoryFile() as memfile:
            with memfile.open(**output_profile) as mem:
                reproject(
                    extracted_data,
                    rasterio.band(mem, 1),
                    src_transform=transform,
                    src_crs=crs,
                    src_nodata=attrs["_FillValue"],
                    resampling=Resampling.nearest,
                    warp_mem_limit=0.1,
                    num_threads=8,
                )
                del extracted_data
                overview_level = get_maximum_overview_level(mem.width, mem.height, minsize=tilesize)
                overviews = [2 ** j for j in range(1, overview_level + 1)]
                mem.build_overviews(overviews, Resampling.nearest)
                tags = dict(
                    OVR_RESAMPLING_ALG="NEAREST",
                    kappa0=kappa0,
                    t=t,
                    esun=esun,
                    goes_scene=os.path.basename(src_path),
                )
                mem.update_tags(**tags)
                mem.update_tags(1, **attrs)
                mem._set_all_scales([attrs["scale_factor"]])
                mem._set_all_offsets([attrs["add_offset"]])
                # return the open mem file
                return mem
