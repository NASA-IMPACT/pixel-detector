import fiona
from netCDF4 import Dataset
import numpy as np
#from config import shapefile_path,raw_data_path
from datetime import datetime
import os
from glob import glob
from config import BITMAPS_DIR,WGS84_DIR
from datetime import timedelta
from osgeo import gdal, osr, gdal_array
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyresample import image, geometry, utils,kd_tree,bilinear
import rasterio, rasterio.features
import io_util as io

bands = ['BAND_06']



# def gen_data_from_shapefile(shapefile_path,bands):
#      """Read The Timestamp Of the Shapefiles and decode NC file corresponding to it."""
#      try:
#          print("Load shapefile {}.".format(shapefile_path))
#          with fiona.open(shapefile_path) as shapefile:
#              for feature in shapefile:
#                  geom = feature['geometry']
#                  property = feature['properties']
#                  time_str_start = property['Start']
#                  time_str_end = property['End']
#                  start_time = datetime.strptime(time_str_start,'%Y%j %H%M') + timedelta(hours=5)
#                  end_time = datetime.strptime(time_str_end,'%Y%j %H%M') + timedelta(hours=5)
#                  # Time conversion code goes here
#                  sat_paths = get_bands_path(end_time,end_time,bands,sat_path)

#                  i = 0
#                  for path in band_path:
#                      NDV, xsize, ysize, GeoT, Projection, data = GetnetCDFInfobyName(sat_path+path,'Rad')
#                      data = 0
#                      scaler = MinMaxScaler(feature_range=(0,255))
#                      scaler.fit(data)
#                      data = scaler.transform(data)
#                      outfile = create_geotiff(BITMAPS_DIR+band_array[i], data, NDV, xsize, ysize, GeoT, Projection)
#                      i = i+1
#                      raster_dataset, wgs_location = reproject_dataset(outfile)
#                      raster_dataset = rasterio.open(outfile)
#                      bitmap_image = rasterio.features.rasterize(
#                      [(geom,255)],
#                      out_shape=raster_dataset.shape,
#                      transform=raster_dataset.transform)
#                      io.save_bitmap('test_bmp.bmp', np.asarray(bitmap_image,dtype='uint8'), raster_dataset)
#                      bitmap_image[bitmap_image == 255] = 1
                 
#              return result,bitmap_image



def get_bands_path(start_time,end_time,band_array,sat_path):
    """
        Get the Paths of images matching the band_array keywords, 
        within start, end time
        returns paths array if found, else returns false
    """
    tt_start = start_time.timetuple()
    tt_end = end_time.timetuple()
    hr = tt_start.tm_hour
    yr = tt_start.tm_year
    jl = tt_start.tm_yday
    if tt_start.tm_min >= 15 and int(tt_start.tm_min) <= 45:
        mn = 30
    else:
        mn = 00
    path_list = []

    for band in band_array:
        for fname in glob(sat_path+'*'+str(yr)+'.'+str(jl)+'.'+start_time.strftime('%H')+str(mn)+'??.'+band+'.nc'):
            #for fname in glob(sat_path+'/*'+str(yr)+'.'+str(jl)+'.'+start_time.strftime('%H')+str(mn)+'??.'+band+'.nc'):
                if fname == []:
                    return False
                else:
                    path_list.append(fname)

    return path_list


def band_list(loc,band_array):
    """

    """
    path_list = []

    for band in band_array:
        fname = glob(loc+'*'+band+'*.nc')
        fname = fname[0]
        if fname == []:
            return False
        else:
            path_list.append(fname)

    return path_list


def create_geotiff(suffix, Array, NDV, xsize, ysize, GeoT, Projection): 
    '''
    Creates new GeoTiff from array 
    '''
    DataType = gdal_array.NumericTypeCodeToGDALTypeCode(Array.dtype) 
    if type(DataType)!=np.int: 
        if DataType.startswith('gdal.GDT_')==False: 
            DataType=eval('gdal.GDT_'+DataType)

    NewFileName = suffix +'wgs84.tif' 
    zsize=1
    # create a driver
    driver = gdal.GetDriverByName('GTiff')
    # Set nans to the original No Data Value
    Array[np.isnan(Array)] = NDV
    # Set up the dataset with zsize bands
    DataSet = driver.Create( NewFileName, xsize, ysize, zsize, DataType)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection.ExportToWkt() )
    # Write each slice of the array along the zsize
    DataSet.GetRasterBand(1).WriteArray( np.flipud(Array) ) 
    DataSet.GetRasterBand(1).SetNoDataValue(NDV) 

    DataSet.FlushCache()

    return NewFileName 

def GetnetCDFInfobyName(in_filename, var_name):

    """
        Function to read the original file's projection
    """
    # Open netCDF file
    src_ds = gdal.Open(in_filename)
    if src_ds is None:
        print "Open failed"
        return 0

    if src_ds.GetSubDatasets() > 1:

        # If exists more than one var in the NetCDF...
        subdataset = 'NETCDF:"'+ in_filename + '":' + var_name
        src_ds_sd = gdal.Open(subdataset)
        # begin to read info of the named variable (i.e.,subdataset)
        NDV = src_ds_sd.GetRasterBand(1).GetNoDataValue()
        xsize = src_ds_sd.RasterXSize
        ysize = src_ds_sd.RasterYSize
        GeoT = src_ds_sd.GetGeoTransform()
        Projection = osr.SpatialReference()
        Projection.ImportFromWkt(src_ds_sd.GetProjectionRef())
        # Close the subdataset and the whole dataset
        src_ds_sd = None
        src_ds = None
        # read data using xrray
        xr_ensemble = xr.open_dataset(in_filename)
        data = xr_ensemble[var_name]
        data = np.ma.masked_array(data, mask=data==NDV,fill_value=NDV)
        return NDV, xsize, ysize, GeoT, Projection, data


def convert_raw_to_grayscale(data,band):
    '''
    convert raw data to Effective Tb 
    '''
    if band == 1:
        m = 0.6120196 
        b = -17.749
        k = 0.00189544

    else:
        t_act = None

        # Radiance is needed for all calculations so calculate it first
        m4 = 5.5297     
        b4 = 16.5892
        radiance = (data - b4) / m4 * 10.  # [W/(m2 sr um)]
        c1 = 1.19100e-5      # [mW/m2 sr cm-4]
        c2 = 1.438833        # [K/cm-1]
        v4a = 749.83           # cm-1
        alpha4a =-0.134801
        beta4a = 1.000482
        # Effective temperature in (K)
        t_eff = (c2 * v4a) / np.log(1 + (c1 * v4a**3) / (radiance / 10.))

        # Convert to actual temperature (K)
        t_act = alpha4a + beta4a * t_eff
        return t_act

def resample_data_to_geotiff(lat, lon, Tb, tif_location, area_extent = (-124.6, 29.4, -74.9, 48.2),bilin=True):
    """
    Resample data from given values.
    lon: Longitude value
    lat: Latitude value
    Tb: Value from file
    lat_array: latitude values for bounding box
    lon_array: longitude values for bounding box
    """
    area_id = 'test'
    description = 'test'
    proj_id = 'test'
    projection = '+proj=longlat +ellps=WGS84 +datum=WGS84'
    x_size = 1358
    y_size = 632
    #area_extent = (lon_array[1], lat_array[1], lon_array[0], lat_array[0])
    area_def = utils.get_area_def(area_id,description,proj_id,projection,x_size,y_size,area_extent)
    lons, lats = area_def.get_lonlats()
    swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
    swath_con = image.ImageContainerNearest(Tb,
    swath_def,
    radius_of_influence=100000)
    if ~bilin:
        result = kd_tree.resample_gauss(swath_def, Tb, area_def, neighbours=5,
                                        radius_of_influence=10000, sigmas=25000, nprocs=4)
    elif bilin:
        result = bilinear.resample_bilinear(Tb, swath_def, area_def, radius=50e3,
                neighbours=32, nprocs=2, epsilon=0)


    driver = gdal.GetDriverByName('GTiff')
    ny, nx = np.shape(result)
    ds = driver.Create(tif_location, nx, ny, 1)
    ulx = np.min(lon)
    dx = (np.max(lon) - np.min(lon)) / nx
    dy = (np.max(lat) - np.min(lat)) / ny
    uly = np.max(lat)
    geot = (ulx, dx, 0, uly, 0, -1 * dy)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    ds.SetGeoTransform(geot)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(result)
    ds = None

    return result

