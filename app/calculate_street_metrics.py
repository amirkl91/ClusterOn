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
from libpysal import graph
from libpysal import graph
from packaging.version import Version
import numpy as np
local_crs = "EPSG:2039"

def generate_streets_metrics(streets):
    #TODO: replace it with the working code in notebook once ready

    streets_geometry = streets['geometry']
    street_network_graph = momepy.gdf_to_nx(streets)

    streets['orientation'] = momepy.orientation(streets_geometry)
    streets['longest_axis_length'] = momepy.longest_axis_length(streets_geometry)

    # todo : check if it is related to street length
    streets['compactness_weighted_axis'] = momepy.compactness_weighted_axis(streets_geometry)
    streets['linearity'] = momepy.linearity(streets_geometry)
    queen_graph = libpysal.graph.Graph.build_contiguity(streets_geometry).assign_self_weight()
    streets['alignment'] = momepy.alignment(streets['orientation'] ,queen_graph)
    streets['cell_alignment'] = momepy.cell_alignment(streets['orientation'], osm_graph)
    streets["length"] = streets.length
    streets["linearity"] = momepy.linearity(streets)

    #Calculates natural continuity and hierarchy of street networks
    coins = momepy.COINS(streets)
    stroke_gdf = coins.stroke_gdf()
    stroke_attr = coins.stroke_attribute()
    streets['continuity'] = coins.continuity
    streets['hierarchy'] = coins.hierarchy

    # apperantly every node has only one neighbor:
    # streets['neighbors'] = momepy.neighbors(streets_geometry, queen_graph)

    street_network_graph = momepy.gdf_to_nx(streets)#street graph metrics:

    nx.set_node_attributes(street_network_graph, dict(street_network_graph.degree()), 'degree')
    # cds_length_result = momepy.cds_length(street_network_graph)

    # # Check if the lengths match
    # print(f"Length of cds_length result: {len(cds_length_result)}")
    # print(f"Length of streets GeoDataFrame: {len(streets)}")

    # Make sure they match before assignment
    # if len(cds_length_result) == len(streets):
    #     streets['cds_length'] = cds_length_result
    # else:
    #     print("Mismatch in lengths. Check graph and GeoDataFrame.")
    # Calculates the shortest-path betweenness centrality for nodes.
    # print(momepy.betweenness_centrality(street_network_graph))
    # streets['betweenness_centrality'] = momepy.betweenness_centrality(street_network_graph)

    #Calculates length of cul-de-sacs for subgraph around each node if radius is set, or for whole graph
    cds_lengths = momepy.cds_length(street_network_graph, radius = 5)
    nodes_gdf = osmnx .graph_to_gdfs(street_network_graph, nodes=True, edges=False)

    # Add cds_lengths to the intersections GeoDataFrame
    nodes_gdf['cds_length'] = nodes_gdf.index.map(cds_lengths)

   