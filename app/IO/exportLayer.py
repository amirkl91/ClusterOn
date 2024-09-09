import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from osgeo import ogr
import os
import pyogrio


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

    # Define path for the shapefile, the shp file is saved into the path with the path name
    shp_path = '/Users/annarubtsov/Desktop/shp_name'

    # Export to shapefile
    dataframe_to_shp(gdf, shp_path)