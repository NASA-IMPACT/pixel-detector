import rasterio, os
from glob import glob
from PIL import Image

def png2geotif(png_dir, geotif_dir):
    src_tif_files = glob(f'{geotif_dir}/*.tif')    
    for src_tif_file in src_tif_files:   
        src = rasterio.open(src_tif_file)
        dataset = rasterio.open(os.path.join(f'{png_dir}', src_tif_file.split('/')[-1].replace('.tif', '.png')))
        bands = [1, 2, 3]
        data = dataset.read(bands)
        transform = src.transform
        crs = {'init': 'epsg:4326'}

        with rasterio.open(os.path.join(f'{png_dir}', src_tif_file.split('/')[-1]), 'w', driver='GTiff',
                           width=src.width, height=src.height,
                           count=3, dtype='uint8', nodata=0,
                           transform=transform, crs=crs) as dst:
            dst.write(data.squeeze(), indexes=bands)
            
            
def resize_to_256(images):
    for img in images:
        with Image.open(img) as image:
            image = image.resize((256, 256), Image.ANTIALIAS)
            image.save(img)