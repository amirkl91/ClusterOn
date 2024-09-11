import warnings
import geopandas as gpd
import libpysal
import momepy
import osmnx
import os
import configparser
import numpy as np

cities = ['Jerusalem', 'Tel Aviv', 'Haifa', 'Rishon LeZion', 'Petah Tikva', 'Ashdod',
           'Netanya', 'Beer Sheva', 'Bnei Brak', 'Holon', 'Ramat Gan', 'Rehovot', 'Ashkelon',
             'Bat Yam', 'Beit Shemesh', 'Kfar Saba', 'Herzliya', 'Hadera', 'Modiin', 'Nazareth']

# add CRS
cities = {k: "EPSG:2039" for k in cities}

def calculate_city_data(cities, config_file='config.ini'):
    """Fetches and calculates buildings, streets, and tessellation data for each city."""
    
    # Read or create configuration file
    config = configparser.ConfigParser()
    
    if not os.path.isfile(config_file):
        config['Paths'] = {
            'gdb_folder': '/path/to/your/gdb/folder',
            'output_dir': '/path/to/your/output/folder'
        }
        with open(config_file, 'w') as configfile:
            config.write(configfile)
        print(f"Created configuration file '{config_file}'.")
    
    config.read(config_file)
    gdb_folder = config['Paths']['gdb_folder']
    output_dir = config['Paths']['output_dir']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionaries to store data
    buildings_data = {}
    streets_data = {}
    tessellation_data = {}

    # Loop through each city
    for city, local_crs in cities.items():
        print(f"Processing {city}")

        # File paths
        buildings_file = os.path.join(output_dir, f"{city}_buildings.geojson")
        tessellation_file = os.path.join(output_dir, f"{city}_tessellation.geojson")
        streets_file = os.path.join(output_dir, f"{city}_streets.geojson")

        # Load or fetch and process data
        if os.path.exists(buildings_file) and os.path.exists(tessellation_file) and os.path.exists(streets_file):
            print(f"Loading saved data for {city}...")
            buildings = gpd.read_file(buildings_file)
            tessellation = gpd.read_file(tessellation_file)
            streets = gpd.read_file(streets_file)
        else:
            print(f"Fetching and processing data for {city}...")
            # Fetch building and street data
            buildings = osmnx.geometries.geometries_from_place(city, tags={'building': True})
            buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)
            buildings = buildings[["geometry"]].to_crs(local_crs)
            buildings["uID"] = range(len(buildings))

            osm_graph = osmnx.graph_from_place(city, network_type='drive')
            osm_graph = osmnx.projection.project_graph(osm_graph, to_crs=local_crs)

            streets = osmnx.graph_to_gdfs(osm_graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            streets = momepy.remove_false_nodes(streets)
            streets = streets[["geometry"]]
            streets["nID"] = range(len(streets))

            # Generate tessellation
            limit = momepy.buffered_limit(buildings, 100)
            tessellation = momepy.Tessellation(buildings, "uID", limit, verbose=False, segment=1).tessellation

            buildings = buildings.sjoin_nearest(streets, max_distance=1000, how="left")
            buildings = buildings.drop_duplicates("uID").drop(columns="index_right")
            tessellation = tessellation.merge(buildings[['uID', 'nID']], on='uID', how='left')

            # Save data
            print(f"Saving data for {city}...")
            buildings.to_file(buildings_file, driver="GeoJSON")
            tessellation.to_file(tessellation_file, driver="GeoJSON")
            streets.to_file(streets_file, driver="GeoJSON")

        # Store data in dictionaries
        buildings_data[city] = buildings
        streets_data[city] = streets
        tessellation_data[city] = tessellation

    return buildings_data, streets_data, tessellation_data, gdb_folder, output_dir

def perform_metric(buildings_data, streets_data, tessellation_data, output_dir, metric_function, metric_name):
    """Performs a generic metric calculation based on the provided data for each city."""

    # create output directories
    dir_to_make = os.path.join(output_dir, metric_name)
    if not os.path.exists(dir_to_make):
            os.makedirs(dir_to_make)

    # Dictionary to store metrics for each city
    city_metric_dfs = {}

    # Calculate metrics for each city
    for city in buildings_data.keys():
        print(f"Calculating metrics for {city}...")

        buildings = buildings_data[city]
        tessellation = tessellation_data[city]
        streets = streets_data[city]  # If needed for some metrics

        # Perform custom metric calculation using the passed function
        city_metric_df = metric_function(buildings, streets, tessellation)

        # Store the DataFrame for the city
        city_metric_dfs[city] = city_metric_df

        # Save DataFrame as CSV
        output_file = os.path.join(dir_to_make, f"{city}_{metric_name}.csv")
        city_metric_df.to_csv(output_file, index=False)
        print(f"Saved metrics for {city} to {output_file}")

    return city_metric_dfs

#TODO: maybe to add .copy() when return df
def calculate_neighbor_distance_Q1(buildings, streets, tessellation):
    """Calculates neighbor distance (log) for buildings."""
    queen_1 = libpysal.weights.contiguity.Queen.from_dataframe(tessellation, ids="uID", silence_warnings=True)
    buildings["neighbor_distance"] = momepy.NeighborDistance(buildings, queen_1, "uID", verbose=False).series
    buildings["log_neighbor_distance"] = np.log(buildings["neighbor_distance"] + 1) / np.log(10)
    return buildings[["uID", "neighbor_distance", "log_neighbor_distance"]]

def calculate_neighbor_distance_Q3(buildings, streets, tessellation):
    queen_1 = libpysal.weights.contiguity.Queen.from_dataframe(tessellation, ids="uID", silence_warnings=True)
    # Create a higher-order spatial weights matrix for tessellation (considering neighbors up to 3rd order)
    queen_3 = momepy.sw_high(k=3, weights=queen_1)
    buildings['neighbor_distance_q3'] = momepy.NeighborDistance(buildings, queen_3, "uID", verbose=False).series
    return buildings[["uID", "neighbor_distance_q3"]] 

def calculate_shared_walls_length(buildings, streets, tessellation):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buildings["shared_walls_length"] = momepy.SharedWalls(buildings).series
    return buildings[["uID", "shared_walls_length"]]

def calculate_CoveredArea_Q1(buildings, streets, tessellation):
    queen_1 = libpysal.weights.contiguity.Queen.from_dataframe(tessellation, ids="uID", silence_warnings=True)
    tessellation["covered_area"] = momepy.CoveredArea(tessellation, queen_3, "uID", verbose=False).series
    return tessellation[["uID", "covered_area"]]

# Example usage
if __name__ == "__main__":
    buildings_data, streets_data, tessellation_data, gdb_folder, output_dir = calculate_city_data(cities)
    perform_metric(buildings_data, streets_data, tessellation_data, output_dir, calculate_neighbor_distance, "neighbor_distance_Q1_example")
