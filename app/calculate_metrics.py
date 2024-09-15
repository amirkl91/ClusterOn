import geopandas as gpd
import fiona
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import configparser
import momepy
import libpysal
import pandas as pd
import osmnx
import pandas
from bokeh.plotting import show
from clustergram import Clustergram
from shapely.geometry import Point
import app.helper_functions as helper_functions
from libpysal import graph
from packaging.version import Version
import numpy as np

local_crs = "EPSG:2039"


def generate_graph_metrics(buildings, streets, tessellation, coplanar='raise',knnA=15, knnB=5):
    '''
    :param buildings:
    :param streets:
    :param tessellation:
    :param coplanar: switch to clique if there's overlapping polygons
    :return:
    '''
    delaunay = graph.Graph.build_triangulation(buildings.centroid, coplanar='clique').assign_self_weight()
    blg_orient = momepy.orientation(buildings)
    buildings['alignment'] = momepy.alignment(blg_orient, delaunay)

    knn_1 = graph.Graph.build_knn(buildings.centroid, k=knnA, coplanar='clique')  # adjust k if needed
    contiguity = graph.Graph.build_contiguity(buildings)
    buildings['adjacency'] = momepy.building_adjacency(contiguity, knn_1)  # TODO: check for queen1 and queen3
    buildings['mean_interbuilding_distance'] = momepy.mean_interbuilding_distance(buildings, delaunay, knn_1)  # TODO: check for queen1 and queen3
    buildings['neighbour_distance'] = momepy.neighbor_distance(buildings, delaunay)
    buildings['courtyards_num'] = momepy.courtyards(buildings,
                                                    contiguity)  # Calculate the number of courtyards within the joined structure

    knn_2 = graph.Graph.build_knn(buildings.centroid, k=knnB)
    for col in buildings.columns:
        if pd.api.types.is_numeric_dtype(buildings[col]):
            new_columns = {}
            print(f'Calculating metrics for {col}')
            buildings[f'{col}_shannon'] = momepy.shannon(buildings[col], knn_2)
            buildings[f'{col}_simpson'] = momepy.simpson(buildings[col], knn_2)
            buildings[f'{col}_theil'] = momepy.theil(buildings[col], knn_2)
            buildings[f'{col}_values_range'] = momepy.values_range(buildings[col], knn_2)
            buildings[f'{col}_mean_deviation'] = momepy.mean_deviation(buildings[col], knn_2)

    delaunay = graph.Graph.build_triangulation(buildings.centroid, coplanar='clique').assign_self_weight()
    blg_orient = momepy.orientation(buildings)
    buildings['alignment'] = momepy.alignment(blg_orient, delaunay)

    knn_1 = graph.Graph.build_knn(buildings.centroid, k=15, coplanar='clique')  # adjust k if needed
    contiguity = graph.Graph.build_contiguity(buildings)
    buildings['adjacency'] = momepy.building_adjacency(contiguity, knn_1)  # TODO: check for queen1 and queen3
    buildings['mean_interbuilding_distance'] = momepy.mean_interbuilding_distance(buildings, delaunay,
                                                                                  knn_1)  # TODO: check for queen1 and queen3
    buildings['neighbour_distance'] = momepy.neighbor_distance(buildings, delaunay)
    # Calculate the number of courtyards within the joined structure
    buildings['courtyards_num'] = momepy.courtyards(buildings, contiguity)

    knn_2 = graph.Graph.build_knn(buildings.centroid, k=5)
    for col in buildings.columns:
        if pd.api.types.is_numeric_dtype(buildings[col]):
            new_columns = {}
            print(f'Calculating metrics for {col}')
            buildings[f'{col}_shannon'] = momepy.shannon(buildings[col], knn_2)
            buildings[f'{col}_simpson'] = momepy.simpson(buildings[col], knn_2)
            buildings[f'{col}_theil'] = momepy.theil(buildings[col], knn_2)
            buildings[f'{col}_values_range'] = momepy.values_range(buildings[col], knn_2)
            buildings[f'{col}_mean_deviation'] = momepy.mean_deviation(buildings[col], knn_2)

            # Concatenate the new columns to the buildings DataFrame all at once
            buildings = pd.concat([buildings, pd.DataFrame(new_columns)], axis=1)
            buildings = buildings.copy()

    tess_orient = momepy.orientation(tessellation)
    buildings['cell_orientation'] = momepy.cell_alignment(blg_orient, tess_orient)
    buildings['num_of_neighbours'] = momepy.neighbors(tessellation, contiguity, weighted=True)

    gdf_network = momepy.gdf_to_nx(streets)
    # gdf_network =   # the street network
    buildings['node_density'] = momepy.node_density(gdf_network, buildings)
    buildings['street_profile'] = momepy.street_profile(gdf_network, buildings)

    blg_orient = momepy.orientation(buildings)
    str_orient = momepy.orientation(streets)
    buildings["street_index"] = momepy.get_nearest_street(buildings, streets)
    buildings['street_alignment'] = momepy.street_alignment(blg_orient, str_orient, buildings["street_index"])

    return buildings, streets

def generate_building_metrics(buildings: gpd.geodataframe, height_column_name=None):
    if height_column_name:
        buildings['Area'] = buildings.geometry.area
        buildings['floor_area'] = momepy.floor_area(buildings['Area'], buildings['building_height'])
        buildings['volume'] = momepy.volume(buildings['Area'], buildings['building_height'])
        buildings['form_factor'] = momepy.form_factor(buildings, buildings['building_height'])

    # Check if all geometries are either Polygon or MultiPolygon
    if (buildings.geometry.geom_type == 'Polygon').all():
        buildings['corners'] = momepy.corners(buildings)
        buildings['squareness'] = momepy.squareness(buildings)
        # buildings_centroid_corner_distance = momepy.centroid_corner_distance(buildings)
        # buildings['centroid_corner_distance_mean'] = buildings_centroid_corner_distance['mean']
        # buildings['centroid_corner_distance_std'] = buildings_centroid_corner_distance['std']
    # else:
        # buildings['corners'] = momepy.corners(buildings, include_interiors=True)
        # buildings['squareness'] = momepy.squareness(buildings, include_interiors=True)



    # Basic geometric properties
    buildings['perimeter'] = buildings.geometry.length
    buildings['shape_index'] = momepy.shape_index(buildings, momepy.longest_axis_length(buildings))
    buildings['circular_compactness'] = momepy.circular_compactness(buildings)
    buildings['square_compactness'] = momepy.square_compactness(buildings)
    buildings['weighted_axis_compactness'] = momepy.compactness_weighted_axis(buildings)
    buildings['convexity'] = momepy.convexity(buildings)
    buildings['courtyard_area'] = momepy.courtyard_area(buildings)
    buildings['courtyard_index'] = momepy.courtyard_index(buildings)
    buildings['fractal_dimension'] = momepy.fractal_dimension(buildings)
    buildings['facade_ratio'] = momepy.facade_ratio(buildings)

    # More complex morphological metrics
    buildings['orientation'] = momepy.orientation(buildings)
    buildings['longest_axis_length'] = momepy.longest_axis_length(buildings)
    buildings['equivalent_rectangular_index'] = momepy.equivalent_rectangular_index(buildings)
    buildings['elongation'] = momepy.elongation(buildings)
    buildings['rectangularity'] = momepy.rectangularity(buildings)
    buildings['shared_walls_length'] = momepy.shared_walls(buildings) / buildings['perimeter']
    buildings['perimeter_wall'] = momepy.perimeter_wall(buildings)  # TODO: check for possible parameters
    buildings['perimeter_wall'] = buildings['perimeter_wall'].fillna(0)
    # Metrics related to building adjacency and Graph

    return buildings


def generate_all_metrics(buildings, streets, tessellation, height_column_name=None):
    buildings = generate_building_metrics(buildings, height_column_name)
    streets = generate_streets_metrics(streets)
    buildings, streets = generate_graph_metrics(buildings, streets, tessellation)
    return buildings, streets