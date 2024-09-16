from time import time
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm
from calculate_building_metrics import *
from calculate_street_metrics import generate_streets_metrics
from calculate_junction_metrics import generate_junctions_metrics

place = 'Jerusalem'
local_crs = 'EPSG:2039'
network_type = 'drive'

streets = load_roads_from_osm(place, network_type=network_type)
streets, intersections = pp.get_streets(streets=streets, local_crs=local_crs, get_nodes=True)

buildings = load_buildings_from_osm(place)
buildings = pp.get_buildings(buildings=buildings, streets=streets, intersections=intersections, local_crs=local_crs, )

tesselations, enclosures = pp.get_tessellation(buildings=buildings, streets=streets, 
                                        tess_mode='enclosed', clim='adaptive')

try:
    t0 = time()
    building_metrics(buildings)
    print(f'Building only metrics: {time()-t0} s')
except:
    print("Failed computing building metrics")
try:
    t0 = time()
    contiguity = building_graph_metrics(buildings)
    print(f'Building tesselation metrics: {time()-t0} s')
except:
    print('Failed computing graph-based building metrics')
try:
    to = time()
    building_tess_metrics(buildings, tesselations, contiguity)
    print(f'Building tesselation metrics: {time()-t0} s')
except:
    print('Failed computing building-tessellation metrics')

try:
    t0 = time()
    generate_streets_metrics(streets)
    print(f'Street metrics : {time()-t0} s')
except:
    print('Failed computing street-related metrics')

try:
    t0 = time()
    intersectinos, streets = generate_junctions_metrics(streets)
    print(f'Street junction metrics: {time()-t0} s')
except:
    print('Failed computing junction-related metrics')