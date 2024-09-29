import streamlit as st
from stats_generation2 import (
    output_cluster_stats,
    varify_cleaned_gdf,
    analyze_gdf,
    output_global_stats,
)
import pandas as pd
from helper_functions import find_overall_leading_metrics
import plot_funcs
import zipfile
import geopandas as gpd


######################### Session state initialization #########################
# Initialize session state variables
def init_session_state():
    session_keys = [
        "cluster_merged",
        "results",
        "global_results",
        "cluster_column_name",
    ]
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None


init_session_state()


def process_uploaded_zip(uploaded_zip):
    try:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            # List files inside the zip
            zip_file_names = zip_ref.namelist()

            # Filter for GPKG and SHP files
            gpkg_files = [file for file in zip_file_names if file.endswith(".gpkg")]

            # Process GPKG files
            if gpkg_files:
                for gpkg_file in gpkg_files:
                    gdf = gpd.read_file(zip_ref.open(gpkg_file))
                    st.session_state["cluster_merged"] = gdf
                    st.sidebar.write(f"Loaded {gpkg_file}")
            st.sidebar.success("All GPKG files extracted and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"An error occurred while processing the ZIP file: {e}")


# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.markdown("# Part 3: Analysis of city textures 📊")
st.markdown("""
    ## How to analyze the data:
    1. After completing Part 2 (Data Preprocessing), you have two options to proceed:
        - Directly move on to Part 2, as the processed data is already saved.
        - Alternatively, upload the ZIP file generated in Part 2.
    2. Run Analyze Data.
    3. You can explore the morphological characteristics of urban areas through two main options:
        - Show Global Data: provides an overview of the entire dataset.
        - Show Cluster Data: allowing you to dive deeper into specific clusters for more detailed analysis.
""")

######################### upload: #########################

# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")
# Load data from session state if available
cluster_merged = st.session_state.get("cluster_merged")

# If preprocessed data exists, load it by default
if cluster_merged is not None:
    cluster_column_name = "cluster"
    st.sidebar.success("Preprocessed data loaded by default.")
else:
    st.sidebar.warning("Preprocessed data not found. Please upload a file.")

# Always provide option to upload a ZIP file
cluster_column_name = st.sidebar.text_input(
    "Enter the name of the **cluster** column:", value="cluster"
)
st.session_state["cluster_column_name"] = cluster_column_name
uploaded_zip = st.sidebar.file_uploader(
    "Upload ZIP file from Part 2 (Classfication)", type=["zip"]
)

if uploaded_zip:
    process_uploaded_zip(uploaded_zip)

##############################################################


######################### Analysis: #########################
# TODO: use cluster_column_name
# TODO: check cluster column exists!

# Button to trigger analysis
if st.button("Analyze Data"):
    if cluster_merged is not None:
        st.write("Analyzing data...")

        # Run the analysis process
        cluster_merged = varify_cleaned_gdf(
            cluster_merged
        )  # Ensure it's a cleaned GeoDataFrame
        analyzed_gdf, results, global_results = analyze_gdf(
            cluster_merged, "cluster", None
        )

        st.session_state["results"] = results
        st.session_state["global_results"] = global_results
        st.session_state["analyzed_gdf"] = analyzed_gdf
        st.success("Data analysis completed successfully!")
    else:
        st.warning("Please upload data before analyzing.")

results = st.session_state.get("results")
global_results = st.session_state.get("global_results")
analyze_gdf = st.session_state.get("analyzed_gdf")

# Make sure the file is uploaded and the cluster column exists
if (
    results is not None
    and cluster_column_name is not None
    and global_results is not None
):
    # Create tabs for Global Data and Cluster Data
    tab1, tab2 = st.tabs(["Show Global Data", "Show Cluster Data"])

    # Tab 1: Global Data
    with tab1:
        st.subheader("Global Analysis")
        supervised_importances = global_results["supervised_importances"]

        # Add a select box for the user to choose the number of features to display
        features_num = st.selectbox(
            "Select the number of features to display:",
            options=[5, 10, 15, 20],
            index=1,
        )

        plot_funcs.streamlit_plot_top_important_metrics(
            supervised_importances, features_num
        )

        # Display global statistics
        global_stats = global_results["global_stats"]
        global_stats_df = output_global_stats(global_stats)
        st.write("Overall statistics")
        st.write(global_stats_df)

        # display outliers
        plot_funcs.streamlit_plot_outliers(analyze_gdf, results)
    # Tab 2: Cluster Data
    # Create a session state variable to track whether the full data is shown or hidden
    if "show_full_data" not in st.session_state:
        st.session_state.show_full_data = False

    with tab2:
        # Display the available clusters in the dataset
        unique_clusters = results.keys()

        st.subheader("Analysis per specific Cluster")
        selected_cluster = st.selectbox(
            "Select a cluster to analyze:", options=unique_clusters
        )

        if selected_cluster is not None:
            st.write(f"**Cluster: {selected_cluster}**")

            # Get the cluster statistics DataFrame
            stats_df = output_cluster_stats(selected_cluster, results)

            # Sort by flexibility score
            stats_df_sorted = stats_df.sort_values(
                by="Flexibility Score", ascending=False
            )

            # Display the top 5 and bottom 5 metrics
            top_5 = stats_df_sorted.head(5)
            bottom_5 = stats_df_sorted.tail(5)

            # Show the top 5 and bottom 5
            st.write("Top 5 Metrics by Flexibility Score:")
            st.write(top_5)
            st.write("Bottom 5 Metrics by Flexibility Score:")
            st.write(bottom_5)

            # Button to display the full DataFrame
            if st.button("Show full data"):
                st.write("Full Data for Cluster:")
                st.write(stats_df)

else:
    st.warning("Please upload the data and ensure the cluster column exists.")
