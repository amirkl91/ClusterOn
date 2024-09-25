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
import zipfile

# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.sidebar.markdown("# Classification of city textures 🗺️")
# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")

# Option to load existing data
load_existing = st.sidebar.radio("Choose how to load data:", ("Continue preprocess part", "Upload ZIP file"))

if load_existing == "Upload ZIP file":
    # Upload the ZIP file
    uploaded_zip = st.sidebar.file_uploader("Upload the ZIP file from part 1", type=["zip"])

    if uploaded_zip is not None:
        try:
            # Open the uploaded ZIP file
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                # List of expected file names in the zip
                expected_files = ["merged.csv", "metrics_with_percentiles.csv", "standardized.csv", "buildings.csv"]
                
                # Check if all expected files are present in the zip
                zip_file_names = zip_ref.namelist()
                missing_files = [file for file in expected_files if file not in zip_file_names]
                
                if missing_files:
                    st.error(f"Missing files in the ZIP: {', '.join(missing_files)}")
                else:
                    # Extract each CSV file and load into DataFrames
                    merged_df = pd.read_csv(zip_ref.open("merged.csv"))
                    metrics_df = pd.read_csv(zip_ref.open("metrics_with_percentiles.csv"))
                    standardized_df = pd.read_csv(zip_ref.open("standardized.csv"))
                    buildings_df = pd.read_csv(zip_ref.open("buildings.csv"))
                    
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

elif load_existing == "Continue preprocess part":
    # Load data from session state if it exists
    merged = st.session_state.get('merged')
    standardized = st.session_state.get('standardized')
    percentiles = st.session_state.get('metrics_with_percentiles')
    buildings = st.session_state.get('buildings')

    if merged is not None and standardized is not None and percentiles is not None and buildings is not None:
        st.sidebar.success("Loaded data from preprocess part.")
        st.sidebar.write(merged.head())  # Preview the merged DataFrame
    else:
        st.sidebar.warning("You have to run preprocess part first. Please upload a ZIP file.")

num_clusters = st.selectbox("Select number of clusters:", options=range(2, 21), index=5)
# Run classification button
if st.button("Run classification"):
    # Ensure all necessary data is available
    # Slider to choose the number of clusters
    if 'merged' in st.session_state and 'standardized' in st.session_state and 'metrics_with_percentiles' in st.session_state and 'buildings' in st.session_state:
        merged = st.session_state['merged']
        metrics_with_percentiles = st.session_state['metrics_with_percentiles']
        standardized = st.session_state['standardized']
        buildings = st.session_state['buildings']

        st.subheader("Processing Clusters...")
        cgram = get_cgram(standardized, num_clusters)
        urban_types = add_cluster_col(merged, buildings, cgram, num_clusters-1)
        plot_clusters_st(urban_types) #TODO: need to fix plotting
        #TODO: need to check the reslts for other city than Jerusalem
    else:
        st.warning("Please ensure all data is loaded before running classification.")