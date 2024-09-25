import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer
import metrics
import merge_dfs as md
from generate_clusters import get_cgram, add_cluster_col, plot_clusters_st, select_best_num_of_clusters
from data_output import dataframe_to_gdb
import matplotlib.pyplot as plt
import pandas as pd
import momepy
import os
import fiona
import zipfile

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
    st.sidebar.warning("Preprocessed data not found. Please upload a ZIP file.")

# Always provide option to upload a ZIP file
uploaded_file = st.sidebar.file_uploader("Upload the csv clusters file from part 2", type=["csv"])
cluster_column_name = st.sidebar.text_input("Enter the name of the **cluster** column:", value=None)

# Check if a file was uploaded
if uploaded_file is not None and cluster_column_name is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        urban_types = pd.read_csv(uploaded_file)
        
        # Display success message and preview the DataFrame
        st.sidebar.success("File uploaded and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"An error occurred while reading the CSV file: {e}")

##############################################################


######################### Analysis: #########################
#TODO: use cluster_column_name 

# run analysis
