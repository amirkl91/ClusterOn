import warnings

import geopandas as gpd
import libpysal
import momepy
import osmnx
import pandas as pd
import contextily as ctx
import os
import configparser
import numpy as np

from clustergram import Clustergram
import matplotlib.pyplot as plt
from bokeh.io import output_notebook
from bokeh.plotting import show

def load_data(city, local_crs, buildingsPath=None, streetsPath=None):
    try:
        if buildingsPath:
            # Try to load data from the given paths
            buildings = gpd.read_file(buildingsPath)
        else:
            buildings = osmnx.geometries.geometries_from_place(city, tags={'building': True})
            buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

        if  streetsPath:
            streets = gpd.read_file(streetsPath)
        else:
            osm_graph = osmnx.graph_from_place(city, network_type='all')
            osm_graph = osmnx.projection.project_graph(osm_graph, to_crs=local_crs)

            streets = osmnx.graph_to_gdfs(
                osm_graph,
                nodes=False,
                edges=True,
                node_geometry=False,
                fill_edge_geometry=True
            )
            streets = momepy.remove_false_nodes(streets)
            streets = streets[["geometry"]]
        
        # Standardize CRS
        buildings = buildings.to_crs(local_crs)
        streets = streets.to_crs(local_crs)
        
        # Add unique identifier columns
        buildings["uID"] = range(len(buildings))
        streets["nID"] = range(len(streets))

        # Generate tessellation for manually loaded data
        limit = momepy.buffered_limit(buildings, 100)
        tessellation = momepy.Tessellation(buildings, "uID", limit, verbose=False, segment=1)
        tessellation = tessellation.tessellation

        # Assign nearest streets and merge data with tessellation
        buildings = buildings.sjoin_nearest(streets, max_distance=1000, how="left")
        buildings = buildings.drop_duplicates("uID").drop(columns="index_right")
        tessellation = tessellation.merge(buildings[['uID', 'nID']], on='uID', how='left')

        print(f"Successfully loaded OSM data for {city}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    return buildings, streets, tessellation

def plot_shared_walls(city, buildings, streets):
    buildings["shared_walls"] = momepy.SharedWalls(buildings).series
    buildings.plot("shared_walls", figsize=(15, 15), scheme="natural_breaks", legend=True).set_axis_off()
    plt.show()

# Example of running the functions:
# def main():
#     # Example usage: load data for a city and plot shared walls
#     city = "Jerusalem"
#     local_crs = "EPSG:2039"

#     # Paths for local files (if any) - set to None if you want OSM data
#     buildingsPath = None
#     streetsPath = None

#     # Load data
#     buildings, streets, tessellation = load_data(city, local_crs, buildingsPath, streetsPath)

#     print(f"Plotting shared walls for {city}...")
#     plot_shared_walls(city, buildings, streets)
    

    # Example: load costume data from path
    # city = "Jerusalem"
    # local_crs = "EPSG:2039"
    # buildingsPath = "/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/Binyanim_jps_reka.gdb"
    # streetsPath = None

    # # Load data
    # buildings, streets, tessellation = load_data2(city, local_crs, buildingsPath, streetsPath)

    # print(f"Plotting shared walls for {city}...")
    # plot_shared_walls(city, buildings, streets)


# if __name__ == "__main__":
#     main()