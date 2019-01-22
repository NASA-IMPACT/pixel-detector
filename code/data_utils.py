
import numpy as np
import json
import fiona
import os
import glob
from preprocessing import get_res_for_extent

def rasterize_nc_files(shp_path, ncfile_path):
    '''
    use the lat lon info from shpfile to make smaller rasters of the ncfiles

    '''

    output_path = '/nas/rgroup/dsig/smoke/goes_data_qc/geotiff_files'
    padding     = 0.50 # units in coordinates
    bands_str   = ['M3C01','M3C03']


    for shpfile in list(glob.glob(shp_path+'*.shp')):
        print('opening : '+shpfile)
        with fiona.open(shpfile) as shp:

            bounds = list(shp.bounds)

            # add padding:
            bounds[0] -= padding
            bounds[1] -= padding
            bounds[2] += padding
            bounds[3] += padding

            # start_time = shp[0]['properties']['Start']
            # end_time   = shp[0]['properties']['End']
            time_str = shp.name.split('_')[-2]
            date_str = shp[0]['properties']['Start'].split()[0]
            doy_str  = date_str[-3:]
            hr_str   = time_str[:2]
            min_str = time_str[2:]
            shp_str  = shp.name.split('_')[-1]
            res      = get_res_for_extent(bounds)



            for t in ['0','15']:
                for band in bands_str:


                    start_time = int(hr_str)*60 + int(min_str)
                    end_time = start_time + int(t)
                    end_hour, end_minute = end_time // 60, end_time % 60
                    print('{}:{}'.format(end_hour, end_minute))

                    path =  os.path.join(ncfile_path,doy_str,str(end_time))

                    if os.path.exists(path):

                        file_path = glob.glob(path+'/*'+band+'*'+date_str+time_str+'*.nc')

                        print(path+'*'+band+'/*'+date_str+time_str+'*.nc')

                        if file_path:

                            nfile = 'NETCDF:"' + file_path[0] + '":Rad'
                            warp_options = gdal.WarpOptions(
                                format='GTiff',
                                outputType=gdal.GDT_Float32,
                                resampleAlg=5,
                                width=res[0],
                                height=res[1],
                                outputBounds=bounds,
                                dstSRS=osr.SRS_WKT_WGS84
                            )
                            output_loc = os.path.join(output_path,str(date_str+time_str))

                            if not os.path.exists(output_loc):
                                os.makedirs(output_loc)

                            wr = gdal.Warp(os.path.join(output_loc,band+'.tif'), nfile,
                                options=warp_options)
                            wr.FlushCache()

                    else:
                        print(path+' not found, moving on...')



