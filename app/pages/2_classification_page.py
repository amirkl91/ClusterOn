import streamlit as st
from generate_clusters import add_cluster_col, plot_clusters_st, best_davies_bouldin_score, plot_num_of_clusters
import pandas as pd
import os
import zipfile
import tempfile
import geopandas as gpd

######################### Session state initialization #########################
# Initialize session state variables
def init_session_state():
    session_keys = ['rec_list', 'merged', 'standardized', 'buildings', 'urban_types', 'cluster_fig', 'cluster_merged']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

init_session_state()

######################### Helper functions #########################

@st.cache_data
def convert_df(_df):
    return _df.to_csv().encode("utf-8")

def process_uploaded_zip(uploaded_zip):
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            expected_files = ["merged.gpkg", "standardized.csv", "buildings.gpkg"]
            zip_file_names = zip_ref.namelist()
            missing_files = [file for file in expected_files if file not in zip_file_names]

            if missing_files:
                st.error(f"Missing files in the ZIP: {', '.join(missing_files)}")
            else:
                merged_df = gpd.read_file(zip_ref.open("merged.gpkg"))
                standardized_df = pd.read_csv(zip_ref.open("standardized.csv"))
                buildings_df = gpd.read_file(zip_ref.open("buildings.gpkg"))

                st.session_state['merged'] = merged_df
                st.session_state['standardized'] = standardized_df
                st.session_state['buildings'] = buildings_df

                st.sidebar.success("All files extracted and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"An error occurred while processing the ZIP file: {e}")

def recommend_clusters(max_clusters):
    if 'standardized' in st.session_state:
        rec_list = best_davies_bouldin_score(
            st.session_state.get('standardized'),
            model=st.session_state['cluster_model'], 
            standardize=False, 
            min_clusters=1, 
            max_clusters=max_clusters,
            n_init=13, 
            random_state=None,
            repeat=5
        )
        st.session_state.rec_list = rec_list

def show_cluster_plots():
    if 'standardized' in st.session_state:
        cluster_fig, axes = plot_num_of_clusters(
            st.session_state.get('standardized'), 
            model=st.session_state['cluster_model'], 
            standardize=False, 
            min_clusters=1,
            max_clusters=15,
            n_init=13, 
            random_state=None,
        )
        st.session_state.cluster_fig = cluster_fig

def run_classification(clusters_num, model):
    if all(key in st.session_state and not st.session_state[key].empty for key in ['merged', 'standardized', 'buildings']):
        merged = st.session_state['merged']
        standardized = st.session_state['standardized']
        buildings = st.session_state['buildings']

        urban_types, cluster_merged = add_cluster_col(merged, buildings, standardized, clusters_num, model)
        st.session_state['urban_types'] = urban_types
        st.session_state['cluster_merged'] = cluster_merged

    else:
        st.warning("Please ensure all data is loaded before running classification.")

def save_output_files():
    if st.session_state['cluster_merged'] is not None:
        cluster_merged = st.session_state['cluster_merged']

        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                cluster_merged_path = os.path.join(tmpdirname, "clusters.gpkg")
                cluster_merged.to_file(cluster_merged_path, driver='GPKG')

                zip_filename = os.path.join(tmpdirname, "gpkg_files.zip")
                with zipfile.ZipFile(zip_filename, 'w') as zipf:
                    zipf.write(cluster_merged_path, arcname="clusters.gpkg")
                
                with open(zip_filename, "rb") as gf:
                    st.download_button(
                        label="Download zip for anlysis (part 3)",
                        data=gf,
                        file_name="clusters.zip",
                        mime="application/zip"
                    )
                st.success("ZIP file successfully created and ready for download.")
            except Exception as e:
                st.error(f"An error occurred while saving the ZIP file: {e}")

            try:
                cluster_merged_shp_path = os.path.join(tmpdirname, 'clusters.shp.zip')
                cluster_merged.to_file(cluster_merged_shp_path, driver='ESRI Shapefile')

                cluster_merged_zip = os.path.join(tmpdirname, 'clusters.zip')
                with zipfile.ZipFile(cluster_merged_zip, 'w') as mzip:
                    mzip.write(cluster_merged_shp_path, 'clusters.shp.zip')

                with open(cluster_merged_zip, 'rb') as mzf:
                    st.download_button(
                        label='Download clusters of urban types data as .shp',
                        data=mzf,
                        file_name='clusters_shp.zip',
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
recommend, ask_nclusts = st.columns([1, 1])
with ask_nclusts:
    max_nclusts = int(st.text_input('Maximal number of clusters to check', value=15))
    cluster_model = st.selectbox('Clustering method', ['kmeans', 'gmm', 'mini-batch kmeans', 'hierarchical'])
    st.session_state['cluster_model'] = 'minibatchkmeans' if cluster_model == 'mini-batch kmeans' else cluster_model
with recommend:
    if st.button("Recommend the Number of Clusters"):
        if all(st.session_state.get(key) is not None for key in ['merged', 'standardized', 'buildings']):
            recommend_clusters(max_nclusts)
        else:
            st.error("Please load data first")
    # Display the recommendation
    if st.session_state.rec_list is not None:
        st.write(f"Our recommendations for the number of clusters are ranked from most to least preferred: {st.session_state.rec_list}")

with recommend:
    # Show cluster recommendation plots
    if st.button("Show Recommendation Calculation Plots"):
        if all(st.session_state.get(key) is not None for key in ['merged', 'standardized', 'buildings']):
            show_cluster_plots()
        else:
            st.error("Please load data first")
        
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
            st.write(f"You have selected {clusters_num} clusters.")
    except ValueError:
        st.error("Please enter a valid integer for the number of clusters.")

    # Run classification
    if st.button("Run classification"):
        if all(st.session_state.get(key) is not None for key in ['merged', 'standardized', 'buildings']):
            model = st.session_state['cluster_model']
            run_classification(clusters_num, model)
        else:
            st.error("Please load data first")
    if st.session_state.urban_types is not None:
        plot_clusters_st(st.session_state.urban_types)


# Save output files
save_output_files()