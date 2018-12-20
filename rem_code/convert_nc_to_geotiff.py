import gdal
import osr
def conv_nc_to_geotiff(ncfile,geotiff_path,extent=[-149.765109632,-64.407370418,-0.234889169,64.407370030],res = [10848,10848]):

    nfile = 'NETCDF:"'+ncfile+'":Rad'
    
    translate_options = gdal.TranslateOptions(
        outputType = gdal.GDT_Float32,
        format = 'GTiff',
        noData = 0
        )

    tr = gdal.Translate('test.tif',nfile,options = translate_options)
    tr.FlushCache()


    warp_options = gdal.WarpOptions(
        format = 'GTiff',
        outputType = gdal.GDT_Float32,
        resampleAlg = 5,
        outputBounds = extent,
        dstSRS = osr.SRS_WKT_WGS84
        )

                
    wr = gdal.Warp('test.tif',nfile,options = warp_options)
    wr.FlushCache()


import os
import os.path

def conv_folder(path,dest_path,shapefile):
    for root,dir,ncfiles in os.walk(path):
        for ncfile in ncfiles:
            print(ncfile,dir,root)
            if '.nc' in ncfile:
                if not os.path.exists(os.path.join(dest_path,root[-7:])):
                    os.makedirs(os.path.join(dest_path,root[-7:]))
                    
                conv_nc_to_geotiff(os.path.join(root,ncfile),os.path.join(dest_path,root[-7:],ncfile[:-3]+'.tif'))



def gen_data(jsonfile):

    js = open(jsonfile)
    jsondict = json.loads(js.read())
    
