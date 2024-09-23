import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer
import metrics
import merge_dfs as md
from generate_clusters import get_cgram, add_cluster_col, plot_clusters_st
from data_output import dataframe_to_gdb, save_and_download
import matplotlib.pyplot as plt
import pandas as pd
import momepy
import os
import fiona


# Streamlit App Title
st.title("Morphological Analysis Tool")
st.markdown("# Main page 🌐")

st.sidebar.markdown("# Main page 🌐")
# 1. Upload GDB Layer for Buildings Data
add_upload_header = st.sidebar.header("Step 1: Upload GDB File for Buildings Data")
# Add an upload window to the sidebar:
gdb_folder_path = st.sidebar.text_input("Enter the full directory path to the .gdb folder:",
                                           value="/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/קונטור בניינים/commondata/jps_reka.gdb")
if gdb_folder_path:
    try:
        # Check if the specified path exists and is a directory
        if not os.path.exists(gdb_folder_path) or not os.path.isdir(gdb_folder_path):
            st.sidebar.error("The specified path does not exist or is not a valid directory.")
        else:
            # List layers in the GDB folder
            layers = fiona.listlayers(gdb_folder_path)
            st.sidebar.write(f"Layers found in the specified GDB file: {layers}")

            # Select a layer to load
            layer_index = st.sidebar.selectbox("Select a layer to process", range(len(layers)), format_func=lambda x: layers[x])

            if st.sidebar.button("Load Layer"):
                # Load the selected layer
                gdf = load_gdb_layer(gdb_folder_path, layer_index=layer_index)

                # Store gdf in session_state so it persists between reruns
                st.session_state['gdf'] = gdf

                # Show the first few rows of the loaded data
                st.sidebar.write(gdf.head())

    except Exception as e:
        st.error(f"An error occurred: {e}")


# 2. Enter OSM Data Parameters
st.sidebar.header("Step 2: Enter Streets Data from OSM")
place = st.sidebar.text_input("Enter a city name for OSM Streets data", value="Jerusalem")
local_crs = st.sidebar.text_input("Enter Local CRS (e.g., EPSG:2039)", value="EPSG:2039")
network_type = st.sidebar.selectbox("Select Network Type for OSM", ["drive", "walk", "bike"], index=0)



# # Streamlit App Title
# st.title("Morphological Analysis Tool")



# 3. Button to Run the Processing Functionality
if st.button("Run Preprocessing and Clustering"):
    # Check if gdf is in session_state
    if 'gdf' in st.session_state and st.session_state['gdf'] is not None:
        gdf = st.session_state['gdf']
        # Proceed with further processing using the gdf
        st.write(f"Running analysis on {len(gdf)} buildings.")
        # Load street data from OSM
        streets = load_roads_from_osm(place, network_type=network_type)
        # Preprocess Streets and Buildings
        streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
        buildings = gdf  # Load the buildings data from GDB layer
        buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs)
        # Generate tessellation
        tessellations = pp.get_tessellation(buildings=buildings, streets=streets, 
                                            tess_mode='morphometric', clim='adaptive')
        # Generate metrics
        metrics.generate_building_metrics(buildings)
        queen_1 = metrics.generate_tessellation_metrics(tessellations, buildings)
        metrics.generate_streets_metrics(streets)
        queen_3 = metrics.generate_graph_building_metrics(buildings, streets, queen_1)
        junctions, streets = metrics.generate_junctions_metrics(streets)
        # Merge DataFrames
        merged = md.merge_all_metrics(tessellations, buildings, streets, junctions)
        metrics_with_percentiles = md.compute_percentiles(merged, queen_3)
        standardized = md.standardize_df(metrics_with_percentiles)
        st.success("Preprocessing completed!")

        # User inputs for saving paths
        csv_path = st.text_input("Enter the path to save the CSV file:", value="/Users/annarubtsov/Desktop/app_files")
        gdb_path = st.text_input("Enter the path to save the GDB file:", value="/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/קונטור בניינים/commondata/jps_reka.gdb")

        if st.button("Save Files"):
            # Save to CSV
            save_and_download(buildings, csv_path, gdb_path, "builidngs")
    else:
        st.error("Please upload a GDB file and load the data before running.")