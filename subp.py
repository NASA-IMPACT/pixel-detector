import subprocess
import numpy as np


nc_filename  = 'OR_ABI-L1b-RadC-M3C01_G16_s20182300032307_e20182300035080_c20182300035131.nc'


res = (5947,1713)


str1 = 'gdal_translate -ot float32 -scale[[0 255]] -CO COMPRESS=deflate NETCDF:"'+nc_filename+':Rad" '+ 'temp.tif'

str2 = 'gdalwarp'

subprocess.call([str1])

