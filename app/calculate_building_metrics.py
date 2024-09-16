import geopandas as gpd
# import fiona
# import os
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import configparser
import momepy
# import libpysal
# import pandas as pd
# import osmnx
# import pandas
# from bokeh.plotting import show
# from clustergram import Clustergram
# from shapely.geometry import Point
from libpysal import graph
from libpysal import graph
# from packaging.version import Version
# import numpy as np

import helper_functions

def building_metrics(buildings: gpd.geodataframe, height_column_name=None):
    buildings['area'] = buildings.area
    if height_column_name is not None:
        buildings['floor_area'] = momepy.floor_area(buildings['area'], buildings[height_column_name])
        buildings['volume'] = momepy.volume(buildings['area'], buildings[height_column_name])
        buildings['form_factor'] = momepy.form_factor(buildings, buildings[height_column_name])

    # Basic geometric properties
    buildings['perimeter'] = buildings.geometry.length
    buildings['shape_index'] = momepy.shape_index(buildings, momepy.longest_axis_length(buildings))
    buildings['circular_compactness'] = momepy.circular_compactness(buildings)
    buildings['square_compactness'] = momepy.square_compactness(buildings)
    buildings['weighted_axis_compactness'] = momepy.compactness_weighted_axis(buildings)
    buildings['convexity'] = momepy.convexity(buildings)
    buildings['courtyard_area'] = momepy.courtyard_area(buildings)
    buildings['courtyard_index'] = momepy.courtyard_index(buildings)
    buildings['corners'] = momepy.corners(buildings, include_interiors=True)   # include_interiors=False works only for polygons not multipolygons
    buildings['fractal_dimension'] = momepy.fractal_dimension(buildings)
    buildings['facade_ratio'] = momepy.facade_ratio(buildings)
    buildings[['centr_corn_dist_mean', 'centr_corn_dist_std']] = momepy.centroid_corner_distance(buildings)

    # More complex morphological metrics
    buildings['orientation'] = momepy.orientation(buildings)
    buildings['longest_axis_length'] = momepy.longest_axis_length(buildings)
    buildings['equivalent_rectangular_index'] = momepy.equivalent_rectangular_index(buildings)
    buildings['elongation'] = momepy.elongation(buildings)
    buildings['rectangularity'] = momepy.rectangularity(buildings)
    buildings['squareness'] = momepy.squareness(buildings, include_interiors=True)   # without interiors works only for polygons not multipolygons
    buildings['shared_walls_length'] = momepy.shared_walls(buildings) / buildings['perimeter']
    buildings['perimeter_wall'] = momepy.perimeter_wall(buildings)  # TODO: check for possible parameters
    buildings['perimeter_wall'] = buildings['perimeter_wall'].fillna(0)
    
def building_graph_metrics(buildings):
    # Metrics related to building adjacency and Graph

    delaunay = graph.Graph.build_triangulation(buildings.centroid, coplanar='clique').assign_self_weight()
    orientation = momepy.orientation(buildings)
    buildings['alignment'] = momepy.alignment(orientation, delaunay)

    knn15 = graph.Graph.build_knn(buildings.centroid, k=15, coplanar='clique')  # adjust k if needed
    contiguity = graph.Graph.build_contiguity(buildings)
    buildings['adjacency'] = momepy.building_adjacency(contiguity, knn15)  # TODO: check for queen1 and queen3
    buildings['mean_interbuilding_distance'] = momepy.mean_interbuilding_distance(buildings, delaunay,
                                                                                    knn15)  # TODO: check for queen1 and queen3
    buildings['neighbour_distance'] = momepy.neighbor_distance(buildings, delaunay)
    buildings['courtyards_num'] = momepy.courtyards(buildings, contiguity)  # Calculate the number of courtyards within the joined structure
    return contiguity

def building_tess_metrics(buildings, tessellations, contiguity):
    ### Has wierd errors. Dissregard for now
    # metrics related to tessellation
    tess_orient = momepy.orientation(tessellations)
    buildings['cell_orientation'] = momepy.cell_alignment(buildings['orientation'], tess_orient)
    buildings['num_of_neighbours'] = momepy.neighbors(tessellations, contiguity, weighted=True)

if __name__=='__main__':
    from time import time
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

    t0 = time()
    building_metrics(buildings)
    print(f'Building only metrics: {(t1:=time())-t0} s')    
    contiguity = building_graph_metrics(buildings)
    print(f'Building graph metrics: {(t2:=time()) - t1} s')
    # building_tess_metrics(buildings, tesselations, contiguity)
    # print(f'Building tesselation metrics: {time()-t2} s')
