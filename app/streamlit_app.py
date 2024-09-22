import streamlit as st
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm, load_gdb_layer, handle_gdb_upload, handle_zip_upload
import metrics
import merge_dfs as md
from generate_clusters import get_cgram, add_cluster_col, plot_clusters
from data_output import dataframe_to_gdb
import matplotlib.pyplot as plt
import momepy
import os
import fiona
#from clustergram import Clustergram

# Initialize Streamlit session state variables
if 'gdf' not in st.session_state:
    st.session_state.gdf = None

# Streamlit App Title
st.title("Morphological Analysis Tool")

# 1. Upload GDB Layer for Buildings Data
st.header("Step 1: Upload GDB File for Buildings Data")
# uploaded_zip_file = st.file_uploader("Upload the GDB file for buildings data", type="zip")
# Text input for the directory path to the .gdb file
gdb_folder_path = st.text_input("Enter the full directory path to the .gdb folder:", value="/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/קונטור בניינים/commondata/jps_reka.gdb")

if gdb_folder_path:
    try:
        # Check if the specified path exists and is a directory
        if not os.path.exists(gdb_folder_path) or not os.path.isdir(gdb_folder_path):
            st.error("The specified path does not exist or is not a valid directory.")
        else:
            # List layers in the GDB folder
            layers = fiona.listlayers(gdb_folder_path)
            st.write(f"Layers found in the specified GDB file: {layers}")

            # Select a layer to load
            layer_index = st.selectbox("Select a layer to process", range(len(layers)), format_func=lambda x: layers[x])

            if st.button("Load Layer"):
                # Load the selected layer
                st.session_state.gdf = load_gdb_layer(gdb_folder_path, layer_index=layer_index)

                # Show the first few rows of the loaded data
                st.write(st.session_state.gdf.head())

    except Exception as e:
        st.error(f"An error occurred: {e}")

# 2. Enter OSM Data Parameters
st.header("Step 2: Enter Streets Data from OSM")
place = st.text_input("Enter a city name for OSM Streets data", value="Jerusalem")
local_crs = st.text_input("Enter Local CRS (e.g., EPSG:2039)", value="EPSG:2039")
network_type = st.selectbox("Select Network Type for OSM", ["drive", "walk", "bike"], index=0)

# 3. Button to Run the Processing Functionality
if st.button("Run Preprocessing and Clustering"):
    if st.session_state.gdf is not None and place and local_crs:
        st.info("Processing data, this may take a few moments...")
        
        # Load street data from OSM
        streets = load_roads_from_osm(place, network_type=network_type)
        
        # Preprocess Streets and Buildings
        streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
        buildings = st.session_state.gdf  # Load the buildings data from GDB layer
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

        # Example plot code
        st.write("Plotting ...")

        # Sample data
        # Assuming `buildings` is a GeoDataFrame with a column "shared_walls"
        buildings["shared_walls"] = momepy.SharedWalls(buildings).series

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot buildings with shared walls ratio
        buildings.plot(column="shared_walls", scheme="natural_breaks", legend=True, ax=ax)

        # Add a title
        plt.title(f"Shared Walls Length for Jerus", fontsize=16)

        # Remove axis
        ax.set_axis_off()

        # Show plot in Streamlit
        st.pyplot(fig)

        # Save data to a temporary file
        output_file = "/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/מרקמים/commondata/myproject16.gdb", "urban_types"
        # dataframe_to_gdb(buildings, output_file, "urban_types")
        # st.success("Clusters generated and saved locally!")

        # # Provide option to download the file
        # with open(output_file, "rb") as f:
        #     st.download_button(
        #         label="Download GDB with Clusters",
        #         data=f,
        #         file_name="urban_types.gdb",
        #         mime="application/octet-stream"
        #     )
    else:
        st.error("Please upload a GDB file and provide valid parameters!")
