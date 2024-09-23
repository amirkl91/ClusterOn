import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer, load_buildings_from_osm
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
def process_data(place, network_type, local_crs, _buildings_gdf, _streets_gdf):
    if _buildings_gdf is not None:
        # If buildings data is from GDB
        if isinstance(_buildings_gdf, pd.DataFrame):  
            buildings = _buildings_gdf
        else:
            # Load buildings data from OSM
            buildings = load_buildings_from_osm(place)
        buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs)
    else:
        st.error("No buildings data available.")

    if _streets_gdf is not None:
        # If buildings data is from GDB
        if isinstance(_streets_gdf, pd.DataFrame):  
            streets = _streets_gdf
        else:
            # Load buildings data from OSM
            streets = load_roads_from_osm(place, network_type=network_type)
        streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
    else:
        st.error("No buildings data available.")

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

def load_gdb_data(data_source_key, data_type):
    st.sidebar.header(f"Upload {data_type} Data")
    uploaded_file = st.sidebar.file_uploader(f"Upload a zip file containing a {data_type} GDB", type="zip", key=f"{data_source_key}_upload")

    if uploaded_file:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(f"temp_extracted_{data_source_key}")

        extracted_items = os.listdir(f"temp_extracted_{data_source_key}")
        gdb_path = next((os.path.join(f"temp_extracted_{data_source_key}", item) for item in extracted_items if item.endswith(".gdb")), None)

        if gdb_path:
            try:
                if not os.path.exists(gdb_path) or not os.path.isdir(gdb_path):
                    st.sidebar.error(f"The specified path does not exist or is not a valid directory for {data_type} GDB.")
                else:
                    layers = fiona.listlayers(gdb_path)
                    st.sidebar.write(f"Layers found in the {data_type} GDB: {layers}")

                    layer_index = st.sidebar.selectbox(f"Select a {data_type} layer", range(len(layers)), format_func=lambda x: layers[x], key=f"{data_source_key}_layer")

                    if st.sidebar.button(f"Load {data_type} Layer", key=f"{data_source_key}_load"):
                        gdf = load_gdb_layer(gdb_path, layer_index=layer_index)
                        st.session_state[f'{data_source_key}_gdf'] = gdf
                        st.sidebar.write(gdf.head())
            except Exception as e:
                st.error(f"An error occurred: {e}")


def load_osm_data(data_source_key, data_type):
    st.sidebar.header(f"Enter {data_type} Data from OSM")
    place = st.sidebar.text_input(f"Enter a city name for OSM {data_type} data", value="Jerusalem", key=f"{data_source_key}_place")
    local_crs = st.sidebar.text_input(f"Enter Local CRS (e.g., EPSG:2039)", value="EPSG:2039", key=f"{data_source_key}_crs")
    network_type = st.sidebar.selectbox(f"Select Network Type for {data_type}", ["drive", "walk", "bike"], index=0, key=f"{data_source_key}_network")

    if st.sidebar.button(f"Load OSM {data_type} Data", key=f"{data_source_key}_osm_load"):
        st.sidebar.write(f"OSM data fetched")
        # Here you would load the OSM data

    return place, local_crs, network_type

# Streamlit App Title
st.title("Morphological Analysis Tool")
st.markdown("# Main page 🌐")
st.sidebar.markdown("# Main page 🌐")

# Select buildings data source
st.sidebar.header("Choose Buildings Data Source")
bld_data_source = st.sidebar.radio("Select buildings data source:", ("Upload buildings GDB file", "Use buildings OSM data"))

if bld_data_source == "Upload buildings GDB file":
    load_gdb_data("buildings", "buildings")
elif bld_data_source == "Use buildings OSM data":
    place, local_crs, network_type = load_osm_data("buildings", "buildings")
    st.session_state['buildings_data'] = (place, local_crs, network_type)

# Select streets data source
st.sidebar.header("Choose Streets Data Source")
str_data_source = st.sidebar.radio("Select streets data source:", ("Upload streets GDB file", "Use streets OSM data"))

if str_data_source == "Upload streets GDB file":
    load_gdb_data("streets", "streets")
elif str_data_source == "Use streets OSM data":
    place, local_crs, network_type = load_osm_data("streets", "streets")
    st.session_state['streets_data'] = (place, local_crs, network_type)

# 3. Button to Run the Processing Functionality
if st.button("Run Preprocessing and Clustering"):
    buildings_gdf = st.session_state.get('buildings_gdf')
    streets_gdf = st.session_state.get('streets_gdf')

    if buildings_gdf is not None and streets_gdf is not None:
        place = st.session_state.get('buildings_data', (None, None, None))[0]  # Get place from buildings
        network_type = st.session_state.get('buildings_data', (None, None, None))[2]  # Get network type from buildings
        local_crs = st.session_state.get('buildings_data', (None, None, None))[1]  # Get CRS from buildings

        merged, metrics_with_percentiles, standardized = process_data(place, network_type, local_crs, buildings_gdf, streets_gdf)       
        st.success("Preprocessing completed!")
    else:
        st.error("Please load data before running.")


# if st.button("Run Preprocessing and Clustering"):
#     # Check if gdf is in session_state
#     if 'gdf' in st.session_state and st.session_state['gdf'] is not None:
#         gdf = st.session_state['gdf']
#         # Proceed with further processing using the gdf
#         st.write(f"Running analysis on {len(gdf)} buildings.")
#         merged, metrics_with_percentiles, standardized =process_data(place, network_type, local_crs, gdf)       
#         st.success("Preprocessing completed!")
#     else:
#         st.error("Please upload a GDB file and load the data before running.")

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