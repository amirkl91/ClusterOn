import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer
import metrics
import merge_dfs as md
from generate_clusters import add_cluster_col, plot_clusters_st, best_davies_bouldin_score, plot_num_of_clusters
from data_output import dataframe_to_gdb, save_csv, save_gdf_to_gpkg
import matplotlib.pyplot as plt
import pandas as pd
import momepy
import os
import fiona
import zipfile
import tempfile
import geopandas as gpd

@st.cache_data
def convert_df(_df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return _df.to_csv().encode("utf-8")

# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.markdown("# Part 2: Classification of City Textures 🗺️")
# Description paragraph
st.markdown("""
    ## How to Classify City Textures:
    1. After completing Part 1 (Data Preprocessing), you have two options to proceed:
        - Directly move on to Part 2, as the processed data is already saved.
        - Alternatively, upload the ZIP file generated in Part 1.
    2. (Optional) You can request a recommendation for the optimal number of clusters to divide the city.
    3. Run the classification to visualize the results and download the output.
    """)
######################### upload: #########################

# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")
# Load data from session state if available
merged = st.session_state.get('merged')
standardized = st.session_state.get('standardized')
percentiles = st.session_state.get('metrics_with_percentiles')
buildings = st.session_state.get('buildings')

# If preprocessed data exists, load it by default
if merged is not None and standardized is not None and percentiles is not None and buildings is not None:
    st.sidebar.success("Preprocessed data loaded by default.")
else:
    st.sidebar.warning("Preprocessed data not found. Please upload a ZIP file.")

# Always provide option to upload a ZIP file
uploaded_zip = st.sidebar.file_uploader("Upload ZIP file from Part 1 (Data Preprocessing)", type=["zip"])

if uploaded_zip is not None:
    try:
        # Open the uploaded ZIP file
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            # List of expected file names in the zip
            expected_files = ["merged.gpkg", "metrics_with_percentiles.csv", "standardized.csv", "buildings.gpkg"]
            
            # Check if all expected files are present in the zip
            zip_file_names = zip_ref.namelist()
            missing_files = [file for file in expected_files if file not in zip_file_names]
            
            if missing_files:
                st.error(f"Missing files in the ZIP: {', '.join(missing_files)}")
            else:
                # Extract each file and load into DataFrames
                merged_df = gpd.read_file(zip_ref.open("merged.gpkg"))  # Use GeoPandas for GPKG files
                metrics_df = pd.read_csv(zip_ref.open("metrics_with_percentiles.csv"))
                standardized_df = pd.read_csv(zip_ref.open("standardized.csv"))
                buildings_df = gpd.read_file(zip_ref.open("buildings.gpkg"))  # Use GeoPandas for GPKG files

                # Store in session state for use in the next step
                st.session_state['merged'] = merged_df
                st.session_state['metrics_with_percentiles'] = metrics_df
                st.session_state['standardized'] = standardized_df
                st.session_state['buildings'] = buildings_df

                # Show a success message and preview one of the DataFrames
                st.sidebar.success("All files extracted and loaded successfully!")
                st.sidebar.write(merged_df.head())  # Preview the merged DataFrame
    except Exception as e:
        st.sidebar.error(f"An error occurred while processing the ZIP file: {e}")
##############################################################


######################### clusters: #########################

# recommend clusters
if st.button("Recommend the Number of Clusters"):
    if 'standardized' in st.session_state:
        rec_list = best_davies_bouldin_score(st.session_state.get('standardized'), model='kmeans', standardize=False, min_clusters=1,
                     max_clusters=15,
                     n_init=13, random_state=42,repeat=5)
        st.write(f"Our recommendations for the number of clusters are ranked from most to least preferred: {rec_list}")

if st.button("Show Recommendation Calculation Plots"):
    if 'standardized' in st.session_state:
        cluster_fig, axes = plot_num_of_clusters(st.session_state.get('standardized'), model='kmeans', standardize=False, min_clusters=1,
                                        max_clusters=15,
                                        n_init=13, random_state=42)
        # Show plot in Streamlit
        st.pyplot(cluster_fig)
        
# Input number of clusters from an unrestricted text box
clusters_num = st.text_input("Enter the number of clusters:", value="7")
try:
    # Convert the input to an integer
    clusters_num = int(clusters_num)
    
    if clusters_num < 1:
        st.warning("Please enter a number greater than 0.")
    else:
        st.success(f"You have selected {clusters_num} clusters.")
        
except ValueError:
    st.error("Please enter a valid integer for the number of clusters.")
# Run classification button
if st.button("Run classification"):
    # Ensure all necessary data is available
    # Slider to choose the number of clusters
    if 'merged' in st.session_state and 'standardized' in st.session_state and 'metrics_with_percentiles' in st.session_state and 'buildings' in st.session_state:
        merged = st.session_state['merged']
        metrics_with_percentiles = st.session_state['metrics_with_percentiles']
        standardized = st.session_state['standardized']
        buildings = st.session_state['buildings']

        urban_types = add_cluster_col(merged, buildings, standardized, clusters_num)
        st.session_state['urban_types'] = urban_types
        plot_clusters_st(urban_types)
    else:
        st.warning("Please ensure all data is loaded before running classification.")

##############################################################


######################### save: #########################

if 'urban_types' in st.session_state:
    urban_types = st.session_state['urban_types']

    with tempfile.TemporaryDirectory() as tmpdirname:       
        try:
        # Create a temporary directory
            # Define file paths for each gpkg file
            urban_types_path = os.path.join(tmpdirname, "urban_types.gpkg")
            
            # Convert DataFrames to GPKG and save
            merged.to_file(urban_types_path, driver='GPKG')

            # Create a ZIP file containing all the CSVs
            zip_filename = os.path.join(tmpdirname, "gpkg_files.zip")
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                zipf.write(urban_types_path, arcname="urban_types.gpkg")
            
            # Provide download link for the ZIP file
            with open(zip_filename, "rb") as gf:
                st.download_button(
                    label="Download GPKG of urban types",
                    data=gf,
                    file_name="urban_types.zip",
                    mime="application/zip"
                )

            st.success("ZIP file successfully created and ready for download.")
        except Exception as e:
            st.error(f"An error occurred while saving the ZIP file: {e}")

        try:
            # save to shp
            urban_types_shp_path = os.path.join(tmpdirname, 'urban_types.shp.zip')
            urban_types.to_file(urban_types_shp_path, driver='ESRI Shapefile')
            urban_types_zip = os.path.join(tmpdirname, 'urban_types.zip')
            with zipfile.ZipFile(urban_types_zip, 'w') as mzip:
                mzip.write(urban_types_shp_path, 'urban_types.shp.zip')
            with open(urban_types_zip, 'rb') as mzf:
                st.download_button(
                    label='Download urban types data as .shp',
                    data = mzf,
                    file_name='urban_types_shp.zip',
                    mime='application/zip'
                )
        except Exception as e:
            st.error(f"An error occurred while saving: {e}")
else:
    merged = None
    st.warning("Please upload files first, then run the preprocess.")

##################################################
