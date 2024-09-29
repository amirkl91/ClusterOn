import streamlit as st
import preprocess as pp
from data_input import load_roads_from_osm, load_gdb_layer, load_buildings_from_osm
import metrics
import merge_dfs as md
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import momepy
import geopandas as gpd
import os
import fiona
import zipfile
from PIL import Image
import tempfile

# Initialize session state variables
def initialize_session_state():
    session_keys = ['buildings', 'streets', 'place', 'crs', 'buildings_gdf', 'streets_gdf', 'c', 'cluster_column_name', 'computed_metrics', 'merged', 'metrics_with_percentiles', 'standardized', 'metrics_zip', 'buildings_uploaded', 'streets_uploaded']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None


@st.cache_data
def convert_df(_df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return _df.to_csv().encode("utf-8")

# @st.cache_data
def return_osm_params(session_string):
    return st.session_state.get(session_string)[0], st.session_state.get(session_string)[1], st.session_state.get(session_string)[2]

@st.cache_data
def process_data(place, network_type, local_crs, _buildings_gdf, _streets_gdf, height_column_name, cluster_column_name, user_selections):
    if _buildings_gdf is not None:
        # If buildings data is from GDB
        if isinstance(_buildings_gdf, pd.DataFrame):  
            buildings = _buildings_gdf
    else:
        # Load buildings data from OSM
        buildings = load_buildings_from_osm(place)
    
    if _streets_gdf is not None:
        # If buildings data is from GDB
        if isinstance(_streets_gdf, pd.DataFrame):  
            streets = _streets_gdf
    else:
        # Load buildings data from OSM
        streets = load_roads_from_osm(place, network_type=network_type)
    
    streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
    junctions, streets = metrics.generate_junctions_metrics(streets, user_selections, selected=user_selections)
    buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, 
                                 local_crs=local_crs, height_name=height_column_name, cluster_name=cluster_column_name)
    # Generate tessellation
    tessellations = pp.get_tessellation(buildings=buildings, streets=streets, 
                                        tess_mode='morphometric', clim='adaptive')

    # Generate metrics
    metrics.generate_building_metrics(buildings, height_column_name=height_column_name, selected=user_selections)
    queen_1 = metrics.generate_tessellation_metrics(tessellations, buildings, selected=user_selections)
    metrics.generate_streets_metrics(streets, selected=user_selections)
    queen_3 = metrics.generate_graph_building_metrics(buildings, streets, queen_1, selected=user_selections)

    # Merge DataFrames
    merged = md.merge_all_metrics(tessellations, buildings, streets, junctions)
    metrics_with_percentiles = md.compute_percentiles(merged, queen_3)
    standardized = md.standardize_df(metrics_with_percentiles)

    st.session_state['buildings'] = buildings
    st.session_state['merged'] = merged
    st.session_state['metrics_with_percentiles'] = metrics_with_percentiles
    st.session_state['standardized'] = standardized
    st.session_state['streets'] = streets

    return merged, metrics_with_percentiles, standardized, buildings, streets

def load_gdb_data(data_source_key, data_type):
    st.sidebar.header(f"Upload {data_type} Data")
    uploaded_file = st.sidebar.file_uploader(f"Upload a zip file containing a {data_type} GDB", type="zip", key=f"{data_source_key}_upload")

    if uploaded_file is not None:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(f"temp_extracted_{data_source_key}")

        extracted_items = os.listdir(f"temp_extracted_{data_source_key}")
        # Check for GDB or Shapefile
        gdb_path = next((os.path.join(f"temp_extracted_{data_source_key}", item) for item in extracted_items if item.endswith(".gdb")), None)
        shp_path = next((os.path.join(f"temp_extracted_{data_source_key}", item) for item in extracted_items if item.endswith(".shp")), None)

        # Print all the extracted files
        extract_dir = f"temp_extracted_{data_source_key}"
        st.write(f"Files in {extract_dir}:")
        for item in extracted_items:
            st.write(item)  # This will print all the files in the directory

        # if gdb_path:
        #     try:
        #         if not os.path.exists(gdb_path) or not os.path.isdir(gdb_path):
        #             st.sidebar.error(f"The specified path does not exist or is not a valid directory for {data_type} GDB.")
        #         else:
        #             layers = fiona.listlayers(gdb_path)
        #             st.sidebar.write(f"Layers found in the {data_type} GDB: {layers}")

        #             layer_index = st.sidebar.selectbox(f"Select a {data_type} layer", range(len(layers)), format_func=lambda x: layers[x], key=f"{data_source_key}_layer")

        #             if st.sidebar.button(f"Load {data_type} Layer", key=f"{data_source_key}_load"):
        #                 gdf = load_gdb_layer(gdb_path, layer_index=layer_index)
        #                 st.session_state[f'{data_source_key}_gdf'] = gdf
        #                 st.sidebar.write(gdf.head())
        #                 if data_type == 'buildings' :
        #                     st.session_state['buildings_uploaded'] = True
        #                 else: 
        #                    st.session_state['streets_uploaded'] = True 
        #     except Exception as e:
        #         st.error(f"An error occurred: {e}")
        # Handle GDB upload
         # Handle Shapefile upload
        if shp_path:
            st.write("We in shp")
            st.write(gdb_path)
            st.write(shp_path)

            try:
                # Load the shapefile into a GeoDataFrame
                gdf = gpd.read_file(shp_path)
                st.session_state[f'{data_source_key}_gdf'] = gdf
                st.sidebar.write(f"{data_type} shapefile loaded successfully!")
                st.sidebar.write(gdf.head())
                
                # Track upload status
                if data_type == 'buildings':
                    st.session_state['buildings_uploaded'] = True
                else:
                    st.session_state['streets_uploaded'] = True
            except Exception as e:
                st.error(f"An error occurred while reading the shapefile: {e}")

        elif gdb_path:
            st.write("We in gdb")
            st.write(gdb_path)
            try:
                if not os.path.exists(gdb_path) or not os.path.isdir(gdb_path):
                    st.sidebar.error(f"The specified path does not exist or is not a valid directory for {data_type} GDB.")
                else:
                    # List layers in GDB
                    layers = fiona.listlayers(gdb_path)
                    st.sidebar.write(f"Layers found in the {data_type} GDB: {layers}")

                    # Select layer
                    layer_index = st.sidebar.selectbox(f"Select a {data_type} layer", range(len(layers)), format_func=lambda x: layers[x], key=f"{data_source_key}_layer")

                    if st.sidebar.button(f"Load {data_type} Layer", key=f"{data_source_key}_load"):
                        gdf = gpd.read_file(gdb_path, layer=layers[layer_index])
                        st.session_state[f'{data_source_key}_gdf'] = gdf
                        st.sidebar.write(gdf.head())

                        # Track upload status
                        if data_type == 'buildings':
                            st.session_state['buildings_uploaded'] = True
                        else:
                            st.session_state['streets_uploaded'] = True 
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.sidebar.error(f"No valid GDB or Shapefile found in the uploaded zip for {data_type} data.")

       
def load_osm_data(data_source_key, data_type, is_streets):
    st.sidebar.header(f"Enter {data_type} Data from OSM")
    default_place = st.session_state['place'] if 'place' in st.session_state else 'Jerusalem'
    place = st.sidebar.text_input(f"Enter a city name for OSM {data_type} data", value=default_place, key=f"{data_source_key}_place")
    st.session_state['place'] = place

    default_crs = st.session_state['crs'] if 'crs' in st.session_state else 'EPSG:2039'
    local_crs = st.sidebar.text_input(f"Enter Local CRS (e.g., EPSG:2039)", value=default_crs, key=f"{data_source_key}_crs")
    st.session_state['crs'] = local_crs
    st.sidebar.markdown("[Don't know your CRS?](https://epsg.io/#google_vignette)", unsafe_allow_html=True)
    if is_streets:
        st.session_state['streets_uploaded'] = True
        network_type = st.sidebar.selectbox(f"Select Network Type for {data_type}", ["drive", "walk", "bike"], index=0, key=f"{data_source_key}_network")
    else:
        st.session_state['buildings_uploaded'] = True
        network_type = None
    if st.sidebar.button(f"Load OSM {data_type} Data", key=f"{data_source_key}_osm_load"):
        st.sidebar.success(f"OSM data fetched")
        if 'zip_filename' in st.session_state:
            del st.session_state['zip_filename']
        if 'metrics_zip' in st.session_state:
            del st.session_state['metrics_zip']

    return place, local_crs, network_type

def display_metrics_window(metrics_list):
    # Use an expander for the metrics selection
    with st.expander("Select Metrics for Analysis ✔️", expanded=False):
        # Create columns to display checkboxes in rows
        num_columns = 3  # Adjust the number of columns as needed
        columns = st.columns(num_columns)

        # Iterate over metrics and display them in columns
        for i, metric in enumerate(metrics_list):
            col_index = i % num_columns  # Determine the column index
            with columns[col_index]:
                is_selected = st.checkbox(f"{metric.capitalize().replace('_',' ')}", value=True)
                
                # Save the user's choice (True/False) in the dictionary
                user_selections[metric] = is_selected

def upload_buildings_data():
    st.sidebar.header("Choose Buildings Data Source")
    bld_data_source = st.sidebar.radio("Select buildings data source:", 
                                        ("Upload buildings GDB file", "Use buildings OSM data"))
    if bld_data_source == "Upload buildings GDB file":
        load_gdb_data("buildings", "buildings")
        height_column_name = st.sidebar.text_input("Enter the name of the **height** column:", value=None)
        cluster_column_name = st.sidebar.text_input("Enter the name of the **cluster** column:", value=None)
        st.session_state['height_column_name'] = height_column_name
        st.session_state['cluster_column_name'] = cluster_column_name
    else:
        place, local_crs, network_type = load_osm_data("buildings", "buildings", False)
        st.session_state['buildings_data'] = (place, local_crs, network_type)
    

def upload_streets_data():
    st.sidebar.header("Choose Streets Data Source")
    str_data_source = st.sidebar.radio("Select streets data source:", 
                                        ("Upload streets GDB file", "Use streets OSM data"))
    if str_data_source == "Upload streets GDB file":
        load_gdb_data("streets", "streets")
    else:
        place, local_crs, network_type = load_osm_data("streets", "streets", True)
        st.session_state['streets_data'] = (place, local_crs, network_type)

def preprocess_and_generate_metrics():
    buildings_gdf = st.session_state.get('buildings_gdf')
    streets_gdf = st.session_state.get('streets_gdf')
    height_column_name = st.session_state.get('height_column_name')
    cluster_column_name = st.session_state.get('cluster_column_name')

    session_string = 'streets_data' if streets_gdf is None else 'buildings_data'
    place, local_crs, network_type = return_osm_params(session_string)
    merged, metrics_with_percentiles, standardized, buildings, streets = process_data(
        place, network_type, local_crs, buildings_gdf, streets_gdf, height_column_name, cluster_column_name, user_selections)
    
    st.success("Preprocessing completed!")
    st.session_state['computed_metrics'] = True
    st.session_state.update({
        'merged': merged,
        'metrics_with_percentiles': metrics_with_percentiles,
        'standardized': standardized,
        'buildings': buildings,
        'streets': streets
    })

def save_processed_data():
    if 'merged' in st.session_state and 'metrics_with_percentiles' in st.session_state:
        merged = st.session_state['merged']
        standardized = st.session_state['standardized']
        buildings = st.session_state['buildings']
        streets = st.session_state['streets']

        save_files = st.checkbox('Prepare files for saving?')
        
        if save_files:
            with tempfile.TemporaryDirectory() as tmpdirname:
                try:
                    zip_filename = zip_checkpoint(tmpdirname, merged, standardized, buildings)
                    with open(zip_filename, "rb") as gf:
                        st.download_button(
                            label="Download zip for classification (part 2)",
                            data=gf,
                            file_name="class_chckpt.zip",
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"An error occurred while saving the ZIP file: {e}")

                try:
                    metrics_zip = zip_data(tmpdirname, streets, buildings)
                    with open(metrics_zip, 'rb') as mzf:
                        st.download_button(
                            label='Download data as .shp',
                            data=mzf,
                            file_name='metrics.zip',
                            mime='application/zip'
                        )
                    st.success("Files successfully created and ready for download.")
                except Exception as e:
                    st.error(f"An error occurred while saving: {e}")
        else:
            merged = None
            st.warning("Please upload files first, then run the preprocess.")

def zip_checkpoint(tempdir, _merged, standardized, _buildings):
    # Define file paths for each gpkg file
    merged_gpkg_path = os.path.join(tempdir, "merged.gpkg")
    standardized_csv_path = os.path.join(tempdir, "standardized.csv")
    buildings_gpkg_path = os.path.join(tempdir, "buildings.gpkg")
    
    # Convert DataFrames to CSV and save them
    _merged.to_file(merged_gpkg_path, driver='GPKG')
    standardized.to_csv(standardized_csv_path, index=False)
    _buildings.to_file(buildings_gpkg_path, driver='GPKG')

    # Create a ZIP file containing all the CSVs
    zip_filename = os.path.join(tempdir, "gpkg_files.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(merged_gpkg_path, arcname="merged.gpkg")
        zipf.write(standardized_csv_path, arcname="standardized.csv")
        zipf.write(buildings_gpkg_path, arcname="buildings.gpkg")
    st.session_state['zip_filename'] = zip_filename
    return zip_filename

def zip_data(tempdir, _streets, _buildings):
    bldg_shp_path = os.path.join(tempdir, 'buildings.shp.zip')
    str_shp_path = os.path.join(tempdir, 'streets.shp.zip')
    _buildings.to_file(bldg_shp_path, driver='ESRI Shapefile')
    _streets.to_file(str_shp_path, driver='ESRI Shapefile')
    metrics_zip = os.path.join(tempdir, 'metrics.zip')
    with zipfile.ZipFile(metrics_zip, 'w') as mzip:
        mzip.write(bldg_shp_path, 'buildings.shp.zip')
        mzip.write(str_shp_path, 'streets.shp.zip')
    st.session_state['metrics_zip'] = metrics_zip
    return metrics_zip

def plot_metric(data_toplot, metric_toplot, streets):
    fig, ax = plt.subplots(figsize=(10,10))
    data_toplot.plot(ax=ax, column=metric_toplot, cmap='Spectral', scheme='naturalbreaks', legend=True, linewidth=1.0)
    ax.set_axis_off()
    ctx.add_basemap(ax=ax, crs=streets.crs, source=ctx.providers.CartoDB.Positron)
    return fig
    
def plot_metrics():
    if 'computed_metrics' in st.session_state:
        bldg_metrics_toplot = ['adjacency', 'alignment', 'Area', 'circular_compactness', 'corners', 
                               'courtyard_area', 'courtyards_num', 'elongation', 'equivalent_rectangular_index', 
                               'facade_ratio', 'fractal_dimension', 'longest_axis_length', 'mean_interbuilding_distance', 
                               'neighbour_distance', 'orientation', 'perimeter', 'rectangularity', 'shape_index', 
                               'shared_walls', 'square_compactness', 'squareness', 'street_alignment', 'closeness', 
                               'degree', 'gamma', 'mean_nd', 'meshedness', 'straightness']

        str_metrics_toplot = ['openness', 'str_length', 'str_linearity', 'str_longest_axis', 'str_orientation', 
                              'width', 'width_dev']

        buildings = st.session_state['buildings']
        streets = st.session_state['streets']
        merged = st.session_state['merged']
        st.header('Plot metrics')
        data_choice = st.radio('Choose data type:', ('Buildings', 'Streets'))

        gdf_to_plot = merged if data_choice == 'Buildings' else streets
        metrics_toplot = bldg_metrics_toplot if data_choice == 'Buildings' else str_metrics_toplot

        metric_toplot = st.selectbox('Select which metric to plot', metrics_toplot)

        if 'metric_to_plot' in st.session_state and st.session_state['metric_to_plot'] == metric_toplot:
            fig_toplot = st.session_state['fig_to_plot']
        else:
            fig_toplot = plot_metric(gdf_to_plot, metric_toplot, streets)
            st.session_state['fig_to_plot'] = fig_toplot

        st.session_state['metric_to_plot'] = metric_toplot
        st.pyplot(fig_toplot)

def display_images():
    image_1 = Image.open("app/app_design/momepyIcon.png")
    image_2 = Image.open("app/app_design/cidrIcon.png")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image_1, use_column_width=True)
    with col3:
        st.image(image_2, use_column_width=True)


# List of metrics
metrics_list = sorted([
    'squareness',
    'perimeter',
    'shape_index',
    'circular_compactness',
    'square_compactness',
    'weighted_axis_compactness',
    'courtyard_area',
    'courtyard_index',
    'fractal_dimension',
    'facade_ratio',
    'orientation',
    'longest_axis_length',
    'equivalent_rectangular_index',
    'elongation',
    'rectangularity',
    'shared_walls',
    'perimeter_wall',
    'alignment',
    'adjacency',
    'mean_interbuilding_distance',
    'neighbour_distance',
    'courtyards_num',
    'street_alignment',
    'tess_area',
    'tess_circ_compactness',
    'tess_convexity',
    'tess_neighbors',
    'tess_covered_area',
    'tess_build_area_raio',  # Corrected typo from 'tess_build_area_raio'
    'tess_orientation',
    'tess_building_orientation',
    'tess_alignment',
    'tess_equivalent_rect_index',
    'mm_len',
    'str_orientation',
    'str_longest_axis',
    'str_linearity',
    'str_length',
    'width',
    'openness',
    'width_dev',
    'degree',
    'closeness',
    'cyclomatic',
    'edge_node_ratio',
    'gamma',
    'mean_nd',
    'meanlen',
    'meshedness',
    'node_density',
    'node_density_weighted',
    'straightness'
])
# Create a dictionary to store user selections
user_selections = {}

######################### Main App Code #########################

st.set_page_config(layout="wide")

st.title("Morphological Analysis Tool 🌍📌📏")
st.markdown("# Part 1: Preprocess 🧹 & Metrics generation 📐")
# Description paragraph
st.markdown("""
    ## Steps to Process Your Data:
    1. Please upload a buildings and streets data or provide OSM parameters.
        If you want to use a .gdb file - **Upload a Zip file** containing your GeoDatabase (.gdb file).
    2. Run the preprocess button.
    3. You will be able to download the processed data.
    """)

datacol, plotcol = st.columns(2)

### Data side of app
with datacol:
    display_metrics_window(metrics_list)

    ######################### upload: #########################
    upload_buildings_data()
    upload_streets_data()
    ##################################################

    ######################### pre-process: #########################
    if st.button("Run preprocessing and generate metrics"):
        if st.session_state.get('buildings_uploaded') and st.session_state.get('streets_uploaded'):
            preprocess_and_generate_metrics()
        else:
            st.error("Please load data first")
    ##################################################

    ######################### save: #########################
    save_processed_data()
    ##################################################

### Plots side of app
with plotcol:
    plot_metrics()
##################################################

### Design
display_images()
