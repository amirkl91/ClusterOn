import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm

place = 'Jerusalem'
local_crs = 'EPSG:2039'
network_type = 'drive'

streets = load_roads_from_osm(place, network_type=network_type)
streets, intersections = pp.get_streets(streets=streets, local_crs=local_crs, get_nodes=True)

buildings = load_buildings_from_osm(place)
buildings = pp.get_buildings(buildings=buildings, streets=streets, intersections=intersections, local_crs=local_crs, )

tesselations, enclosures = pp.get_tessellation(buildings=buildings, streets=streets, 
                                           tess_mode='enclosed', clim='adaptive')
