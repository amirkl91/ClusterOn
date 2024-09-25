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
from PIL import Image
import tempfile

@st.cache_data
def convert_df(_df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return _df.to_csv().encode("utf-8")

@st.cache_data
def return_osm_params(session_string):
    return st.session_state.get(session_string)[0], st.session_state.get(session_string)[1], st.session_state.get(session_string)[2]

@st.cache_data
def process_data(place, network_type, local_crs, _buildings_gdf, _streets_gdf, height_column_name):
    if _buildings_gdf is not None:
        # If buildings data is from GDB
        if isinstance(_buildings_gdf, pd.DataFrame):  
            buildings = _buildings_gdf
    else:
        # Load buildings data from OSM
        buildings = load_buildings_from_osm(place)
    
    if _streets_gdf is not None:
        # If buildings data is from GDB
        if isinstance(_streets_gdf, pd.DataFrame):  
            streets = _streets_gdf
    else:
        # Load buildings data from OSM
        streets = load_roads_from_osm(place, network_type=network_type)
    
    streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
    buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs, height_name=height_column_name)
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

    st.session_state['buildings'] = buildings
    st.session_state['merged'] = merged
    st.session_state['metrics_with_percentiles'] = metrics_with_percentiles
    st.session_state['standardized'] = standardized

    return merged, metrics_with_percentiles, standardized, buildings

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

def load_osm_data(data_source_key, data_type, is_streets):
    st.sidebar.header(f"Enter {data_type} Data from OSM")
    place = st.sidebar.text_input(f"Enter a city name for OSM {data_type} data", value="Jerusalem", key=f"{data_source_key}_place")
    local_crs = st.sidebar.text_input(f"Enter Local CRS (e.g., EPSG:2039)", value="EPSG:2039", key=f"{data_source_key}_crs")
    # Add a hyperlink
    st.sidebar.markdown("[Don't know your CRS?](https://epsg.io/#google_vignette)", unsafe_allow_html=True)
    if is_streets:
        network_type = st.sidebar.selectbox(f"Select Network Type for {data_type}", ["drive", "walk", "bike"], index=0, key=f"{data_source_key}_network")
    else:
        network_type = None
    if st.sidebar.button(f"Load OSM {data_type} Data", key=f"{data_source_key}_osm_load"):
        st.sidebar.write(f"OSM data fetched")
        # Here you would load the OSM data

    return place, local_crs, network_type

# List of metrics
metrics_list = [
    'squareness',
    'perimeter',
    'shape_index',
    'circular_compactness',
    'square_compactness',
    'weighted_axis_compactness',
    'courtyard_area',
    'courtyard_index',
    'fractal_dimension',
    'facade_ratio',
    'orientation',
    'longest_axis_length',
    'equivalent_rectangular_index',
    'elongation',
    'rectangularity',
    'shared_walls_length',
    'perimeter_wall',
    'alignment',
    'adjacency',
    'mean_interbuilding_distance',
    'neighbour_distance',
    'courtyards_num',
    'street_alignment',
    'tess_area',
    'tess_circ_compactness',
    'tess_convexity',
    'tess_neighbors',
    'tess_covered_area',
    'tess_build_area_raio',  # Corrected typo from 'tess_build_area_raio'
    'tess_orientation',
    'tess_building_orientation',
    'tess_alignment',
    'tess_equivalent_rect_index',
    'mm_len',
    'str_orientation',
    'str_longest_axis',
    'str_linearity',
    'str_length',
    'width',
    'openness',
    'width_dev',
    'degree',
    'closeness',
    'cyclomatic',
    'edge_node_ratio',
    'gamma',
    'mean_nd',
    'meanlen',
    'meshedness',
    'node_density',
    'node_density_weighted',
    'straightness'
]

# Create a dictionary to store user selections
user_selections = {}


st.set_page_config(layout="wide")


# Display the session state
# st.write("### Current Session State")
# st.write(st.session_state)

st.title("Morphological Analysis Tool 🌍📌📏")
# Description paragraph
st.markdown("""
    ## Steps to Process Your Data:
    1. Please upload a buildings and streets data or provide OSM parameters.
        If you want to use a .gdb file - **Upload a Zip file** containing your GeoDatabase (.gdb file).
    2. Run the preprocess button.
    3. You will be able to download the processed data.
    """)

# Use an expander for the metrics selection
with st.expander("Select Metrics for Analysis ✔️", expanded=False):
    # Create columns to display checkboxes in rows
    num_columns = 5  # Adjust the number of columns as needed
    columns = st.columns(num_columns)

    # Iterate over metrics and display them in columns
    for i, metric in enumerate(metrics_list):
        col_index = i % num_columns  # Determine the column index
        with columns[col_index]:
            is_selected = st.checkbox(f"{metric.capitalize()}", value=True)
            
            # Save the user's choice (True/False) in the dictionary
            user_selections[metric] = is_selected

st.sidebar.markdown("# Preprocess 🧹 & Metrics generation 📐")

######################### upload: #########################

# Select buildings data source
st.sidebar.header("Choose Buildings Data Source")
bld_data_source = st.sidebar.radio("Select buildings data source:", ("Upload buildings GDB file", "Use buildings OSM data"))

if bld_data_source == "Upload buildings GDB file":
    load_gdb_data("buildings", "buildings")
    height_column_name = st.sidebar.text_input("Enter the name of the **height** column:", value=None)
    st.session_state['height_column_name'] = height_column_name
elif bld_data_source == "Use buildings OSM data":
    place, local_crs, network_type = load_osm_data("buildings", "buildings", False)
    st.session_state['buildings_data'] = (place, local_crs, network_type)

# Select streets data source
st.sidebar.header("Choose Streets Data Source")
str_data_source = st.sidebar.radio("Select streets data source:", ("Upload streets GDB file", "Use streets OSM data"))

if str_data_source == "Upload streets GDB file":
    load_gdb_data("streets", "streets")
elif str_data_source == "Use streets OSM data":
    place, local_crs, network_type = load_osm_data("streets", "streets", True)
    st.session_state['streets_data'] = (place, local_crs, network_type)

##################################################


######################### pre-process: #########################

# 3. Button to Run the Processing Functionality
if st.button("Run preprocessing and generate metrics"):
    # TODO: use the user_selections dictionary before preprocessing
    buildings_gdf = st.session_state.get('buildings_gdf')
    streets_gdf = st.session_state.get('streets_gdf')
    height_column_name = st.session_state.get('height_column_name')
    if streets_gdf is None:
        session_string = 'streets_data'
    elif buildings_gdf is None:
        session_string = 'buildings_data'
    place, local_crs, network_type = return_osm_params(session_string)
    merged, metrics_with_percentiles, standardized, buildings = process_data(place, network_type, local_crs, buildings_gdf, streets_gdf, height_column_name)       
    st.success("Preprocessing completed!")

##################################################

#TODO: still didnt find a solution for saving gdb as zip:
######################### save: #########################

# User inputs for saving paths
gdb_path = st.text_input("Enter the path to save the gdb file:", value="/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/קונטור בניינים/commondata/jps_reka.gdb")
layer_name = st.text_input("Enter layer name to save the gdb file:", value="all_metrics")

# Check if data exists in session state before proceeding
if 'merged' in st.session_state and 'metrics_with_percentiles' in st.session_state and 'standardized' in st.session_state and 'buildings' in st.session_state:
    merged = st.session_state['merged']
    metrics_with_percentiles = st.session_state['metrics_with_percentiles']
    standardized = st.session_state['standardized']
    buildings = st.session_state['buildings']
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Define file paths for each CSV file
            merged_csv_path = os.path.join(tmpdirname, "merged.csv")
            metrics_csv_path = os.path.join(tmpdirname, "metrics_with_percentiles.csv")
            standardized_csv_path = os.path.join(tmpdirname, "standardized.csv")
            buildings_csv_path = os.path.join(tmpdirname, "buildings.csv")
            
            # Convert DataFrames to CSV and save them
            merged.to_csv(merged_csv_path, index=False)
            metrics_with_percentiles.to_csv(metrics_csv_path, index=False)
            standardized.to_csv(standardized_csv_path, index=False)
            buildings.to_csv(buildings_csv_path, index=False)
            
            # Create a ZIP file containing all the CSVs
            zip_filename = os.path.join(tmpdirname, "data_files.zip")
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                zipf.write(merged_csv_path, arcname="merged.csv")
                zipf.write(metrics_csv_path, arcname="metrics_with_percentiles.csv")
                zipf.write(standardized_csv_path, arcname="standardized.csv")
                zipf.write(buildings_csv_path, arcname="buildings.csv")
            
            # Provide download link for the ZIP file
            with open(zip_filename, "rb") as f:
                st.download_button(
                    label="Download All as ZIP for classification step",
                    data=f,
                    file_name="data_files.zip",
                    mime="application/zip"
                )
                
        st.success("ZIP file successfully created and ready for download.")
    except Exception as e:
        st.error(f"An error occurred while saving the ZIP file: {e}")
    
    try:
        # save to CSV
        csv = convert_df(merged)
        save_csv(csv, file_name='buildings.csv')
        # save to gdb
        if st.button("Download gdb"):
            dataframe_to_gdb(merged, gdb_path, layer_name)
            st.success(f"Files successfully saved to {gdb_path}")
    except Exception as e:
        st.error(f"An error occurred while saving: {e}")

else:
    merged = None
    st.warning("Please upload files first, then run the preprocess.")

# if merged is not None:
#     st.write(merged.head())
#     try:
#         # Save to CSV
#         csv = convert_df(merged)
#         save_csv(csv)
#         if st.button("Download gdb"):
#             dataframe_to_gdb(merged, gdb_path, layer_name)
#             st.success(f"Files successfully saved to {gdb_path}")
#     except Exception as e:
#         st.error(f"An error occurred while saving: {e}")
# else:
#     st.warning("Please run the preprocess first.")


##################################################

# Load your images (you can use file paths, URLs, or use file uploader in Streamlit)
image_1 = Image.open("app/app_design/momepyIcon.png")
image_2 = Image.open("app/app_design/cidrIcon.png")
image_3 = Image.open("app/app_design/flatJerus.JPG")

# Create 3 columns
col1, col2, col3 = st.columns(3)
# Display each image in its respective column
with col1:
    st.image(image_1, use_column_width=True)
with col2:
    st.image(image_2, use_column_width=True)
with col3:
    st.image(image_3, use_column_width=True)
