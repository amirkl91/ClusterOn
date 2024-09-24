import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer
import metrics
import merge_dfs as md
from generate_clusters import get_cgram, add_cluster_col, plot_clusters_st
from data_output import dataframe_to_gdb
import matplotlib.pyplot as plt
import pandas as pd
import momepy
import os
import fiona


# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.sidebar.markdown("# Classification of city textures 🗺️")
# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")

 # Display the session state
st.write("### Current Session State")
st.write(st.session_state)

merged = st.session_state.get('merged')
standardized = st.session_state.get('standardized')
percentiles = st.session_state.get('metrics_with_percentiles')
buildings = st.session_state.get('buildings')

if st.button("Run classification"):
   
    st.write(buildings)
    st.write(merged)
    st.write(standardized)
    st.write(percentiles)


    st.subheader("Processing Clusters...")
    cgram = get_cgram(standardized, 4)
    urban_types = add_cluster_col(merged, buildings, cgram, 3)
    plot_clusters_st(urban_types)

### make clusters
        # cgram = get_cgram(standardized, 14)
        # urban_types = add_cluster_col(merged, buildings, cgram, 13)
        # plot_clusters(urban_types)
        # dataframe_to_gdb(urban_types, "/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/מרקמים/commondata/myproject16.gdb", "urban_types")

        # # Example plot code
        # st.write("Plotting ...")
        # # Sample data
        # # Assuming `buildings` is a GeoDataFrame with a column "shared_walls"
        # buildings["shared_walls"] = momepy.SharedWalls(buildings).series
        # # Create a figure and axis
        # fig, ax = plt.subplots(figsize=(15, 15))
        # # Plot buildings with shared walls ratio
        # buildings.plot(column="shared_walls", scheme="natural_breaks", legend=True, ax=ax)
        # # Add a title
        # plt.title(f"Shared Walls Length for Jerus", fontsize=16)
        # # Remove axis
        # ax.set_axis_off()
        # # Show plot in Streamlit
        # st.pyplot(fig)
    # else:
    #     st.error("Please upload a GDB file and load the data before running.")

# File upload for merged metrics
merged_file = st.sidebar.file_uploader("Upload Merged Metrics CSV", type='csv')
if merged_file:
    merged = pd.read_csv(merged_file)

# File upload for buildings data
buildings_file = st.sidebar.file_uploader("Upload Buildings Data CSV", type='csv')
if buildings_file:
    buildings = pd.read_csv(buildings_file)

# Proceed with clustering after files are uploaded
# if 'standardized' in locals() and 'merged' in locals() and 'buildings' in locals():
#     st.success("Files loaded successfully!")
#     # Step 2: Make Clusters
#     st.subheader("Processing Clusters...")
#     cgram = get_cgram(standardized, 4)
#     urban_types = add_cluster_col(merged, buildings, cgram, 3)

#     plot_clusters_st(urban_types)





#dataframe_to_gdb(urban_types, "/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/מרקמים/commondata/myproject16.gdb", "urban_types")


# 1. Upload GDB Layer for Buildings Data
# add_upload_header = st.sidebar.header("Step 1: Upload GDB File for Buildings Data")
# # Add an upload window to the sidebar:
# gdb_folder_path = st.sidebar.text_input("Enter the full directory path to the .gdb folder:",
#                                            value="/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/קונטור בניינים/commondata/jps_reka.gdb")
# if gdb_folder_path:
#     try:
#         # Check if the specified path exists and is a directory
#         if not os.path.exists(gdb_folder_path) or not os.path.isdir(gdb_folder_path):
#             st.sidebar.error("The specified path does not exist or is not a valid directory.")
#         else:
#             # List layers in the GDB folder
#             layers = fiona.listlayers(gdb_folder_path)
#             st.sidebar.write(f"Layers found in the specified GDB file: {layers}")

#             # Select a layer to load
#             layer_index = st.sidebar.selectbox("Select a layer to process", range(len(layers)), format_func=lambda x: layers[x])

#             if st.sidebar.button("Load Layer"):
#                 # Load the selected layer
#                 gdf = load_gdb_layer(gdb_folder_path, layer_index=layer_index)

#                 # Store gdf in session_state so it persists between reruns
#                 st.session_state['gdf'] = gdf

#                 # Show the first few rows of the loaded data
#                 st.sidebar.write(gdf.head())

#     except Exception as e:
#         st.error(f"An error occurred: {e}")


# 2. Enter OSM Data Parameters
# st.sidebar.header("Step 2: Enter Streets Data from OSM")
# place = st.sidebar.text_input("Enter a city name for OSM Streets data", value="Jerusalem")
# local_crs = st.sidebar.text_input("Enter Local CRS (e.g., EPSG:2039)", value="EPSG:2039")
# network_type = st.sidebar.selectbox("Select Network Type for OSM", ["drive", "walk", "bike"], index=0)



# # Streamlit App Title
# st.title("Morphological Analysis Tool")



# 3. Button to Run the Processing Functionality
# if st.button("Run Preprocessing and Clustering"):
#     # Check if gdf is in session_state
#     if 'gdf' in st.session_state and st.session_state['gdf'] is not None:
#         gdf = st.session_state['gdf']
#         # Proceed with further processing using the gdf
#         st.write(f"Running analysis on {len(gdf)} buildings.")
#         # Load street data from OSM
#         streets = load_roads_from_osm(place, network_type=network_type)
#         # Preprocess Streets and Buildings
#         streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
#         buildings = gdf  # Load the buildings data from GDB layer
#         buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs)
#         # Generate tessellation
#         tessellations = pp.get_tessellation(buildings=buildings, streets=streets, 
#                                             tess_mode='morphometric', clim='adaptive')
#         # Generate metrics
#         metrics.generate_building_metrics(buildings)
#         queen_1 = metrics.generate_tessellation_metrics(tessellations, buildings)
#         metrics.generate_streets_metrics(streets)
#         queen_3 = metrics.generate_graph_building_metrics(buildings, streets, queen_1)
#         junctions, streets = metrics.generate_junctions_metrics(streets)
#         # Merge DataFrames
#         merged = md.merge_all_metrics(tessellations, buildings, streets, junctions)
#         metrics_with_percentiles = md.compute_percentiles(merged, queen_3)
#         standardized = md.standardize_df(metrics_with_percentiles)
#         st.success("Preprocessing completed!")

        ### make clusters
        # cgram = get_cgram(standardized, 14)
        # urban_types = add_cluster_col(merged, buildings, cgram, 13)
        # plot_clusters(urban_types)
        # dataframe_to_gdb(urban_types, "/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/מרקמים/commondata/myproject16.gdb", "urban_types")

        # # Example plot code
        # st.write("Plotting ...")
        # # Sample data
        # # Assuming `buildings` is a GeoDataFrame with a column "shared_walls"
        # buildings["shared_walls"] = momepy.SharedWalls(buildings).series
        # # Create a figure and axis
        # fig, ax = plt.subplots(figsize=(15, 15))
        # # Plot buildings with shared walls ratio
        # buildings.plot(column="shared_walls", scheme="natural_breaks", legend=True, ax=ax)
        # # Add a title
        # plt.title(f"Shared Walls Length for Jerus", fontsize=16)
        # # Remove axis
        # ax.set_axis_off()
        # # Show plot in Streamlit
        # st.pyplot(fig)
    # else:
    #     st.error("Please upload a GDB file and load the data before running.")