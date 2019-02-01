# pixel-detector
A pixel based classifier that uses GOES raw products and shapefiles for generating truth set.


## Training

- The training of the model can be do- ne by running `code/train.py`.
- Accepts model and data configurations from `code/config.json`
  - `config.json` is formatted as follows:
  ```
  {
    "type"			    : <'pixel' or 'DeConv'>,
    "num_neighbor"	: <n for n*n neighborhood of pixel to predict >,
    "jsonfile"		  : "<location of json containing data information to be used for training>",
    "num_epoch"		  : <number of epochs>,
    "batch_size"	  : <batch size (10000)>,
    "model_path"	  : <path/to/keras/model>,
    "eval_json"		  : <location of json containing data information to be used for evaluation>,
    "pred_json"		  : <location of json containing data information to be used for prediction>
  }
  ```
  - The json file in `jsonfile` contains information about data. It needs to be formatted as follows:
 
  ```
   [
   {
     "ncfile": <path to the GOES 16 nc File eg:`.../2018/143/23/`>,
     "nctime": <time string in 'yyyyqqqhhmm' format. This should be a part of the ncfile name>,
     "shp":    <`path/to/shp/file` shapefile denoting smoke occurence in the `ncfile` (in WGS84 Coordinates) >,
     "extent": <extent information in lat,lon eg:[-110.0,33.0,-105.0,37.0]>,
     "start": <unused for now>,
     "end":   <unused for now>
  }, ...
  ]
  ```
 - After training is finished, the trained model is stored in `model_path` from `config.json`
 - To make subsequent training and evaluation faster, the raster products obtained after transforming information 
 from geo projection to WGS84 (shapefile projection) are cached in location given by `TIFF_DIR`. The code uses this 
 cache until the files are deleted manually.
 
 
## Prediction
### GeoJson Prediction
- Prediction on the model can be done by calling `infer.py`
- the `Predicter` class in `infer.py` requires: `<path/to/ncfile>`,`datetime.datetime object`, `extent`, `path/to/model`
to initialize
- Once initialized, The `Predicter.predict` method returns a GeoJson Dictionary object containing the predicted 
smoke plumes

### Batch Prediction
- ...
## Evaluation
- ...
