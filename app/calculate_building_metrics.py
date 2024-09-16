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

local_crs = "EPSG:2039"



def get_all_building_metrics(buildings: gpd.geodataframe, height_column_name=None):
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
    # buildings['corners'] = momepy.corners(buildings, include_interiors=True)   # include_interiors=False works only for polygons not multipolygons
    buildings['fractal_dimension'] = momepy.fractal_dimension(buildings)
    buildings['facade_ratio'] = momepy.facade_ratio(buildings)
    # buildings_centroid_corner_distance = momepy.centroid_corner_distance(buildings)
    # buildings['centroid_corner_distance_mean'] = buildings_centroid_corner_distance['mean']
    # buildings['centroid_corner_distance_std'] = buildings_centroid_corner_distance['std']
    # More complex morphological metrics
    buildings['orientation'] = momepy.orientation(buildings)
    buildings['longest_axis_length'] = momepy.longest_axis_length(buildings)
    buildings['equivalent_rectangular_index'] = momepy.equivalent_rectangular_index(buildings)
    buildings['elongation'] = momepy.elongation(buildings)
    # buildings['linearity'] = momepy.linearity(buildings) # does not work on buildings just on streets
    buildings['rectangularity'] = momepy.rectangularity(buildings)
    # buildings['squareness'] = momepy.squareness(buildings, include_interiors=True)   # without interiors works only for polygons not multipolygons
    buildings['shared_walls_length'] = momepy.shared_walls(buildings) / buildings['perimeter']
    buildings['perimeter_wall'] = momepy.perimeter_wall(buildings)  # TODO: check for possible parameters
    buildings['perimeter_wall'] = buildings['perimeter_wall'].fillna(0)
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
    buildings['courtyards_num'] = momepy.courtyards(buildings,
                                                    contiguity)  # Calculate the number of courtyards within the joined structure

    # metrics related to tessellation
    limit = helper_functions.buffered_limit(buildings, buffer='adaptive')
    tessellation = momepy.morphological_tessellation(buildings, clip=limit)  # need adjustment
    blg_orient = momepy.orientation(buildings)
    tess_orient = momepy.orientation(tessellation)
    buildings['cell_orientation'] = momepy.cell_alignment(blg_orient, tess_orient)
    buildings['num_of_neighbours'] = momepy.neighbors(tessellation, contiguity, weighted=True)


if __name__=='__main__':
    from data_input import load_buildings_from_osm
    from preprocess import get_buildings

    place = 'Jerusalem'
    local_crs = 'EPSG:2039'
    network_type = 'drive'

    buildings = load_buildings_from_osm(place)
    buildings = get_buildings(buildings=buildings, local_crs=local_crs, )

    # get_all_building_metrics(buildings, height_column_name='height')