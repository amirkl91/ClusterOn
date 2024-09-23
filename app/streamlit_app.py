import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer
import metrics
import merge_dfs as md
from generate_clusters import get_cgram, add_cluster_col, plot_clusters_st
from data_output import dataframe_to_gdb, save_csv
import matplotlib.pyplot as plt
import pandas as pd
import momepy
import os
import fiona
import zipfile

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache_data
def process_data(place, network_type, local_crs, _gdf):
    # Load street data from OSM
    streets = load_roads_from_osm(place, network_type=network_type)
    # Preprocess Streets and Buildings
    streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
    buildings = _gdf  # Load the buildings data from GDB layer
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

    st.session_state['merged'] = merged
    st.session_state['metrics_with_percentiles'] = metrics_with_percentiles
    st.session_state['standardized'] = standardized

    return merged, metrics_with_percentiles, standardized

# Streamlit App Title
st.title("Morphological Analysis Tool")
st.markdown("# Main page 🌐")

st.sidebar.markdown("# Main page 🌐")
# 1. Upload GDB Layer for Buildings Data
# Upload zip file containing GDB
uploaded_file = st.sidebar.file_uploader("Upload a zip file containing a GDB", type="zip")

if uploaded_file:
    # Create an in-memory ZipFile object from the uploaded file
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        # Extract the contents of the zip file to a temporary directory
        zip_ref.extractall("temp_extracted")

    # Check the first level for the .gdb folder
    extracted_items = os.listdir("temp_extracted")  # List top-level contents
    gdb_folder_path = next((os.path.join("temp_extracted", item) for item in extracted_items if item.endswith(".gdb")), None)

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

# 3. Button to Run the Processing Functionality
if st.button("Run Preprocessing and Clustering"):
    # Check if gdf is in session_state
    if 'gdf' in st.session_state and st.session_state['gdf'] is not None:
        gdf = st.session_state['gdf']
        # Proceed with further processing using the gdf
        st.write(f"Running analysis on {len(gdf)} buildings.")
        merged, metrics_with_percentiles, standardized =process_data(place, network_type, local_crs, gdf)       
        st.success("Preprocessing completed!")
    else:
        st.error("Please upload a GDB file and load the data before running.")

######## save: #########

# Check if 'merged' exists in session state before using it
if 'merged' in st.session_state:
    merged = st.session_state['merged']
else:
    merged = None
    st.warning("Please run the preprocess step first.")

# User inputs for saving paths
gdb_path = st.text_input("Enter the path to save the gdb file:", value="/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/קונטור בניינים/commondata/jps_reka.gdb")
layer_name = st.text_input("Enter layer name to save the gdb file:", value="mergedApp")

if merged is not None:
    st.write(merged.head())
    try:
        # Save to CSV
        csv = convert_df(merged)
        save_csv(csv)
        if st.button("Download gdb"):
            dataframe_to_gdb(merged, gdb_path, layer_name)
            st.success(f"Files successfully saved to {gdb_path}")
    except Exception as e:
        st.error(f"An error occurred while saving: {e}")
else:
    st.warning("No GeoDataFrame available for saving. Please upload and load the GDB layer.")
########################