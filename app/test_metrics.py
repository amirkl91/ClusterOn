from time import time
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm
import metrics

verbose=True
t0 = time()
place = 'Jerusalem'
local_crs = 'EPSG:2039'
network_type = 'drive'

streets = load_roads_from_osm(place, network_type=network_type)
streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
if verbose: print('Got streets')

buildings = load_buildings_from_osm(place)
buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs, )
if verbose: print('Got buildings')


tessellations = pp.get_tessellation(buildings=buildings, streets=streets, 
                                        tess_mode='morphometric', clim='adaptive')
if verbose: print('Got tessellations')

metrics.generate_building_metrics(buildings)
if verbose: print('Got building metrics')

queen_1 = metrics.generate_tessellation_metrics(tessellations, buildings)
if verbose: print('Got tessellation metrics')

metrics.generate_streets_metrics(streets)
if verbose: print('Got street metrics')

queen_3 = metrics.generate_graph_building_metrics(buildings, streets, queen_1)
if verbose: print('Got graph building metrics')

junctions, streets = metrics.generate_junctions_metrics(streets)
if verbose: print('Got junction metrics')

print(f'Total runtime: {(time()-t0)/60:.2f} mins')
# t0 = time()
# print('Generating building metrics')
# buildings = metrics.generate_building_metrics(buildings)
# print(f'Building metrics: {(t1:=time())-t0:.2f} s')
# print('Generating graph-related building metrics')
# buildings, streets = metrics.generate_graph_metrics(buildings, streets, tessellations)
# print(f'Graph-related building metrics {time()-t1:.2f} s')

# metrics.generate_streets_metrics(streets)
# junctions, streets = metrics.generate_junctions_metrics(streets)