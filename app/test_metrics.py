from time import time
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm
import metrics

place = 'Jerusalem'
local_crs = 'EPSG:2039'
network_type = 'drive'

streets = load_roads_from_osm(place, network_type=network_type)
streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_nodes=True)

buildings = load_buildings_from_osm(place)
buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs, )

tessellations, enclosures = pp.get_tessellation(buildings=buildings, streets=streets, 
                                        tess_mode='enclosed', clim='adaptive')

t0 = time()
print('Generating building metrics')
buildings = metrics.generate_building_metrics(buildings)
print(f'Building metrics: {(t1:=time())-t0:.2f} s')
print('Generating graph-related building metrics')
buildings, streets = metrics.generate_graph_metrics(buildings, streets, tessellations)
print(f'Graph-related building metrics {time()-t1:.2f} s')

metrics.generate_streets_metrics(streets)
junctions, streets = metrics.generate_junctions_metrics(streets)