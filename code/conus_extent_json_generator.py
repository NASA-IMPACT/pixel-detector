import numpy as np
import json
from shapely.geometry import MultiLineString
from shapely.ops import polygonize


def conus_extent_json_generator:
    """
    Takes in lat-long as linear array and generates conus extent json.
    """
    
    lat = np.linspace(-129.17, -65.69, 15)
    long = np.linspace(23.8, 49.38, 15)

    hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(lat[:-1], lat[1:]) for yi in long]
    vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(long[:-1], long[1:]) for xi in lat]

    grids = list(polygonize(MultiLineString(hlines + vlines)))

    poly_bounds = []
    for poly in grids:
        poly_bounds.append(list(poly.bounds))

    json_list = []
    for poly_bound in poly_bounds:
        data = {"ncfile": "/nas/rgroup/dsig/smoke/goes_data_qc/2021/264/22",
            "nctime": "20212642250204",
            "shp": "/nas/rgroup/dsig/smoke/goes_data_qc/shapefiles/hms_smoke20180323_plume3.shp",
            "extent": poly_bound,
            "start": "2018082 2002",
            "end": "2018082 2002"}
        json_list.append(data)

    with open('full_disk.json', 'w') as wf:
        json.dump(json_list, wf, indent=4)
        
if __name__ == '__main__':
    conus_extent_json_generator()