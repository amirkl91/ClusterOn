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


def generate_streets_metrics(streets_gdf):
    streets_geometry = streets_gdf["geometry"]

    streets_gdf["orientation"] = momepy.orientation(streets_geometry)
    streets_gdf["longest_axis_length"] = momepy.longest_axis_length(streets_geometry)
    streets_gdf["compactness_weighted_axis"] = momepy.compactness_weighted_axis(
        streets_geometry
    )
    streets_gdf["linearity"] = momepy.linearity(streets_geometry)
    streets_gdf["length"] = streets_geometry.length
    # Calculates natural continuity and hierarchy of street networks
    coins = momepy.COINS(streets_gdf)
    stroke_attr = coins.stroke_attribute()
    print(stroke_attr.head())
    streets_gdf["stroke_id"] = stroke_attr
    # Group by stroke_id to calculate stroke-level continuity (total length of each stroke)
    stroke_continuity = streets_gdf.groupby("stroke_id")["length"].sum().reset_index()
    stroke_continuity.columns = ["stroke_id", "continuity"]
    # Merge continuity back into the streets GeoDataFrame
    streets_gdf = streets_gdf.merge(stroke_continuity, on="stroke_id", how="left")
    # Rank strokes by length to create a hierarchy
    stroke_continuity["hierarchy"] = stroke_continuity["continuity"].rank(
        ascending=False
    )
    # Merge hierarchy back into the streets GeoDataFrame
    streets_gdf = streets_gdf.merge(
        stroke_continuity[["stroke_id", "hierarchy"]], on="stroke_id", how="left"
    )
    # print(streets_gdf.head())
    return streets_gdf


if __name__ == "__main__":
    place = "Jerusalem, Israel"
    osm_streets = osmnx.graph_from_place(place, network_type="drive")
    # osmnx.plot_graph(osm_streets)
    streets = osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(osm_streets),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    ).reset_index(drop=True)
    streets = streets.drop_duplicates(subset="geometry", keep="first")

    generate_streets_metrics(streets)
