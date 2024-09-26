import streamlit as st
from generate_clusters import add_cluster_col, plot_clusters_st, best_davies_bouldin_score, plot_num_of_clusters
import pandas as pd
import os
import zipfile
import tempfile
import geopandas as gpd

######################### Session state initialization #########################
def init_session_state():
    if 'rec_list' not in st.session_state:
        st.session_state.rec_list = None
    if 'merged' not in st.session_state:
        st.session_state['merged'] = None
    if 'metrics_with_percentiles' not in st.session_state:
        st.session_state['metrics_with_percentiles'] = None
    if 'standardized' not in st.session_state:
        st.session_state['standardized'] = None
    if 'buildings' not in st.session_state:
        st.session_state['buildings'] = None
    if 'urban_types' not in st.session_state:
        st.session_state['urban_types'] = None
    if 'cluster_fig' not in st.session_state:
        st.session_state['cluster_fig'] = None
init_session_state()
######################### Helper functions #########################

@st.cache_data
def convert_df(_df):
    return _df.to_csv().encode("utf-8")

def process_uploaded_zip(uploaded_zip):
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            expected_files = ["merged.gpkg", "metrics_with_percentiles.csv", "standardized.csv", "buildings.gpkg"]
            zip_file_names = zip_ref.namelist()
            missing_files = [file for file in expected_files if file not in zip_file_names]

            if missing_files:
                st.error(f"Missing files in the ZIP: {', '.join(missing_files)}")
            else:
                merged_df = gpd.read_file(zip_ref.open("merged.gpkg"))
                metrics_df = pd.read_csv(zip_ref.open("metrics_with_percentiles.csv"))
                standardized_df = pd.read_csv(zip_ref.open("standardized.csv"))
                buildings_df = gpd.read_file(zip_ref.open("buildings.gpkg"))

                st.session_state['merged'] = merged_df
                st.session_state['metrics_with_percentiles'] = metrics_df
                st.session_state['standardized'] = standardized_df
                st.session_state['buildings'] = buildings_df

                st.sidebar.success("All files extracted and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"An error occurred while processing the ZIP file: {e}")

def recommend_clusters():
    if 'standardized' in st.session_state:
        rec_list = best_davies_bouldin_score(
            st.session_state.get('standardized'),
            model='kmeans', 
            standardize=False, 
            min_clusters=1, 
            max_clusters=15,
            n_init=13, 
            random_state=42,
            repeat=5
        )
        st.session_state.rec_list = rec_list

def show_cluster_plots():
    if 'standardized' in st.session_state:
        cluster_fig, axes = plot_num_of_clusters(
            st.session_state.get('standardized'), 
            model='kmeans', 
            standardize=False, 
            min_clusters=1,
            max_clusters=15,
            n_init=13, 
            random_state=42
        )
        st.session_state.cluster_fig = cluster_fig

def run_classification(clusters_num):
    if all(key in st.session_state and not st.session_state[key].empty for key in ['merged', 'standardized', 'metrics_with_percentiles', 'buildings']):
        merged = st.session_state['merged']
        standardized = st.session_state['standardized']
        buildings = st.session_state['buildings']

        urban_types = add_cluster_col(merged, buildings, standardized, clusters_num)
        st.session_state['urban_types'] = urban_types
    else:
        st.warning("Please ensure all data is loaded before running classification.")

def save_output_files():
    if st.session_state['urban_types'] is not None:
        urban_types = st.session_state['urban_types']

        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                urban_types_path = os.path.join(tmpdirname, "urban_types.gpkg")
                urban_types.to_file(urban_types_path, driver='GPKG')

                zip_filename = os.path.join(tmpdirname, "gpkg_files.zip")
                with zipfile.ZipFile(zip_filename, 'w') as zipf:
                    zipf.write(urban_types_path, arcname="urban_types.gpkg")
                
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
                urban_types_shp_path = os.path.join(tmpdirname, 'urban_types.shp.zip')
                urban_types.to_file(urban_types_shp_path, driver='ESRI Shapefile')

                urban_types_zip = os.path.join(tmpdirname, 'urban_types.zip')
                with zipfile.ZipFile(urban_types_zip, 'w') as mzip:
                    mzip.write(urban_types_shp_path, 'urban_types.shp.zip')

                with open(urban_types_zip, 'rb') as mzf:
                    st.download_button(
                        label='Download urban types data as .shp',
                        data=mzf,
                        file_name='urban_types_shp.zip',
                        mime='application/zip'
                    )
            except Exception as e:
                st.error(f"An error occurred while saving: {e}")
    else:
        st.warning("Please upload files first, then run the preprocess.")

######################### Main App Code #########################
# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.markdown("# Part 2: Classification of City Textures 🗺️")
st.markdown("""
    ## How to Classify City Textures:
    1. After completing Part 1 (Data Preprocessing), you have two options to proceed:
        - Directly move on to Part 2, as the processed data is already saved.
        - Alternatively, upload the ZIP file generated in Part 1.
    2. (Optional) You can request a recommendation for the optimal number of clusters to divide the city.
    3. Run the classification to visualize the results and download the output.
""")

# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP file from Part 1 (Data Preprocessing)", type=["zip"])

if uploaded_zip:
    process_uploaded_zip(uploaded_zip)

# Recommend clusters
if st.button("Recommend the Number of Clusters"):
    recommend_clusters()
# Display the recommendation
if st.session_state.rec_list is not None:
    st.write(f"Our recommendations for the number of clusters are ranked from most to least preferred: {st.session_state.rec_list}")

# Show cluster recommendation plots
if st.button("Show Recommendation Calculation Plots"):
    show_cluster_plots()
# Display the recommendation
if st.session_state.cluster_fig is not None:
    st.pyplot(st.session_state.cluster_fig)

# Input number of clusters
clusters_num = st.text_input("Enter the number of clusters:", value="7")
try:
    clusters_num = int(clusters_num)
    if clusters_num < 1:
        st.warning("Please enter a number greater than 0.")
    else:
        st.success(f"You have selected {clusters_num} clusters.")
except ValueError:
    st.error("Please enter a valid integer for the number of clusters.")

# Run classification
if st.button("Run classification"):
    run_classification(clusters_num)
    if st.session_state.urban_types is not None:
        plot_clusters_st(st.session_state.urban_types)


# Save output files
save_output_files()