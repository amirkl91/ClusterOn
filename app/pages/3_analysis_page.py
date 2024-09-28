import streamlit as st
from stats_generation2 import output_cluster_stats, varify_cleaned_gdf, analyze_gdf
import pandas as pd


# Streamlit App Title
st.title("Morphological Analysis Tool 🌍📌📏")
st.sidebar.markdown("# Analysis of city textures 📊")

######################### upload: #########################

# Sidebar for uploading files
st.sidebar.subheader("Upload Data Files")
# Load data from session state if available
cluster_merged = st.session_state.get('cluster_merged')

# If preprocessed data exists, load it by default
if cluster_merged is not None:
    cluster_column_name = "cluster"
    st.sidebar.success("Preprocessed data loaded by default.")
else:
    st.sidebar.warning("Preprocessed data not found. Please upload a file.")

# Always provide option to upload a ZIP file
uploaded_file = st.sidebar.file_uploader("Upload the csv clusters file from part 2", type=["csv"])
cluster_column_name = st.sidebar.text_input("Enter the name of the **cluster** column:", value="cluster")

# Check if a file was uploaded
if uploaded_file is not None and cluster_column_name is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        cluster_merged = pd.read_csv(uploaded_file)
        st.session_state['cluster_merged'] = cluster_merged
        # Display success message and preview the DataFrame
        st.sidebar.success("File uploaded and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"An error occurred while reading the CSV file: {e}")

##############################################################


######################### Analysis: #########################
#TODO: use cluster_column_name 

# Button to trigger analysis
if st.button("Analyze Data"):
    if cluster_merged is not None:
        st.write("Analyzing data...")

        # Run the analysis process
        cluster_merged = varify_cleaned_gdf(cluster_merged)  # Ensure it's a cleaned GeoDataFrame
        analyzed_gdf, results, global_results = analyze_gdf(cluster_merged, "cluster", None)
        
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

    if selected_cluster is not None:
        st.write(f"**Cluster: {selected_cluster}**")
        # Display cluster statistics (you can customize this function further)
        stats_df = output_cluster_stats(selected_cluster, results)
        # Display some summary statistics (customize as needed)
        st.write(stats_df)
        
else:
    st.warning("Please upload the data and ensure the cluster column exists.")

##############################################################
