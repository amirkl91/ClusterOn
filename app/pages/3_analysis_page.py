import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer
import metrics
import merge_dfs as md
from generate_clusters import add_cluster_col, plot_clusters_st
from data_output import dataframe_to_gdb
from stats_generation2 import output_cluster_stats, varify_cleaned_gdf, analyze_gdf
import matplotlib.pyplot as plt
import pandas as pd
import momepy
import os
import fiona
import zipfile

# Set the page layout to wide
st.set_page_config(layout="wide")

# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.sidebar.markdown("# Analysis of city textures 📊")

######################### upload: #########################

# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")
# Load data from session state if available
urban_types = st.session_state.get('urban_types')

# If preprocessed data exists, load it by default
if urban_types is not None:
    cluster_column_name = "cluster"
    st.sidebar.success("Preprocessed data loaded by default.")
else:
    st.sidebar.warning("Preprocessed data not found. Please upload a file.")

# Always provide option to upload a ZIP file
uploaded_file = st.sidebar.file_uploader("Upload the csv clusters file from part 2", type=["csv"])
cluster_column_name = st.sidebar.text_input("Enter the name of the **cluster** column:", value=None)

# Check if a file was uploaded
if uploaded_file is not None and cluster_column_name is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        urban_types = pd.read_csv(uploaded_file)
        st.session_state['urban_types'] = urban_types
        # Display success message and preview the DataFrame
        st.sidebar.success("File uploaded and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"An error occurred while reading the CSV file: {e}")

##############################################################


######################### Analysis: #########################
#TODO: use cluster_column_name 

# Button to trigger analysis
if st.button("Analyze Data"):
    # Make sure urban_types is available and proceed with analysis
    if urban_types is not None:
        st.write("Analyzing data...")

        # Run the analysis process
        gdf = urban_types
        gdf = varify_cleaned_gdf(gdf)  # Ensure it's a cleaned GeoDataFrame
        analyzed_gdf, results, global_results = analyze_gdf(gdf, "cluster", None)
        
        st.session_state['results'] = results  # Save processed data back to session state
        st.success("Data analysis completed successfully!")
    else:
        st.warning("Please upload data before analyzing.")
   
results = st.session_state.get('results')
# Make sure the file is uploaded and the cluster column exists
if results is not None and cluster_column_name is not None:
    
    # Display the available clusters in the dataset
    unique_clusters = results.keys()
    
    st.subheader("Cluster Selection and Analysis")
    selected_cluster = st.selectbox("Select a cluster to analyze:", options=unique_clusters)

    if selected_cluster:
        st.write(f"**Cluster: {selected_cluster}**")
        # Display cluster statistics (you can customize this function further)
        stats_df = output_cluster_stats(selected_cluster, results)
        # Display some summary statistics (customize as needed)
        st.write(stats_df)
        
else:
    st.warning("Please upload the data and ensure the cluster column exists.")

##############################################################
