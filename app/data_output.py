import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import pyogrio
import streamlit as st
from io import BytesIO


def dataframe_to_shp(dataframe, shp_path):
    """
    Export a DataFrame to a .shp file.

    Parameters:
    - dataframe: pd.DataFrame or gpd.GeoDataFrame
    - shp_path: str, path to the .shp file (including .shp extension)
    """
    # Ensure the DataFrame has geometry data
    if not isinstance(dataframe, gpd.GeoDataFrame):
        # Convert DataFrame to GeoDataFrame if necessary
        if 'geometry' not in dataframe.columns:
            raise ValueError("DataFrame must contain a 'geometry' column")
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(dataframe, geometry='geometry')
    else:
        gdf = dataframe

    # Set the CRS if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=2039, inplace=True)  # Set CRS to WGS84 (EPSG:2039 Israel TM Grid) by default

    # Export to shapefile
    gdf.to_file(shp_path)

    print(f"Data exported to {shp_path}")

def dataframe_to_gdb(dataframe, gdb_path, layer_name):
    """
    Export a DataFrame to a .gdb file.
    NOTE: The GDB file should already exist; this function adds the layer to it.

    Parameters:
    - dataframe: pd.DataFrame or gpd.GeoDataFrame
    - gdb_path: str, path to the .gdb file (including .gdb extension)
    - layer_name: str, name of the layer in the .gdb file
    """
    # Ensure the DataFrame has geometry data
    if not isinstance(dataframe, gpd.GeoDataFrame):
        # Convert DataFrame to GeoDataFrame if necessary
        if 'geometry' not in dataframe.columns:
            raise ValueError("DataFrame must contain a 'geometry' column")
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(dataframe, geometry='geometry')
    else:
        gdf = dataframe

    # Set the CRS if not already set
    if gdf.crs is None:
        gdf.set_crs(epsg=2039, inplace=True)  # Set CRS to WGS84 (EPSG:2039 Israel TM Grid) by default

    # Export to FileGDB
    pyogrio.write_dataframe(gdf, gdb_path, layer=layer_name, driver='OpenFileGDB')
    print(f"Data exported to {gdb_path} with layer name '{layer_name}'")

# A function saves the dataframes and creates download link
def save_csv(csv, file_name):
    # Create download links
    st.download_button(
        label=f"Download CSV",
        data=csv,
        file_name=file_name,
        mime='text/csv'
    )

# Function to save GeoDataFrame as GPKG and return a download link
def save_gdf_to_gpkg(gdf, filename):
    # Create a BytesIO object to store the file in memory
    buffer = BytesIO()
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:2039')  # Adjust based on your data
    # Save the GeoDataFrame to the buffer as a GPKG file
    gdf.to_file(buffer, driver="GPKG", layer="layer_name")
    # Set the buffer position to the start to read from the beginning
    buffer.seek(0)
    
    # Streamlit's download button to download the GPKG file
    st.download_button(
        label="Download GPKG",
        data=buffer,
        file_name=filename,
        mime="application/geopackage+sqlite3"
    )



# Example usage
if __name__ == "__main__":
    # Sample DataFrame with geometry
    df = pd.DataFrame({
        'name': ['Location1', 'Location2'],
        'latitude': [34.05, 40.71],
        'longitude': [-118.24, -74.01]
    })

    # Convert to GeoDataFrame
    df['geometry'] = df.apply(lambda x: Point((x['longitude'], x['latitude'])), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    #shp example
    #Define path for the shapefile, the shp file is saved into the path with the name <shp_name>
    shp_path = '/Users/annarubtsov/Desktop/shp_name'
    # Export to shapefile
    dataframe_to_shp(gdf, shp_path)

    # gdb example
    gdb_path = '/Users/annarubtsov/Desktop/myproject16.gdb'
    layer_name = 'layerName'
    # Export to FileGDB
    dataframe_to_gdb(gdf, gdb_path, layer_name)