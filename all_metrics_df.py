from time import time
from app import preprocess as pp
from app import data_input
from app import metrics
import pandas as pd
import geopandas as gpd
import os

# Cities and their CRS
cities = ['Jerusalem', 'Tel Aviv', 'Haifa', 'Rishon LeZion', 'Petah Tikva', 'Ashdod',
           'Netanya', 'Beer Sheva', 'Bnei Brak', 'Holon', 'Ramat Gan', 'Rehovot', 'Ashkelon',
           'Bat Yam', 'Beit Shemesh', 'Kfar Saba', 'Herzliya', 'Hadera', 'Modiin', 'Nazareth']

crs_mapping = {city: "EPSG:2039" for city in cities}  # Mapping CRS to cities

def process_city(city, network_type='drive', config_file='config.ini'):
    # Read or create configuration file
    config = configparser.ConfigParser()
    
    if not os.path.isfile(config_file):
        config['Paths'] = {
            'gdb_folder': '/path/to/your/gdb/folder',
            'output_dir': '/path/to/your/output/folder',
            'output_folder': '/path/to/your/output/folder'
        }
        with open(config_file, 'w') as configfile:
            config.write(configfile)
        print(f"Created configuration file '{config_file}'.")
    
    config.read(config_file)
    output_folder = config['Paths']['output_folder']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    local_crs = crs_mapping[city]
    
    # Load roads and buildings
    streets = data_input.load_roads_from_osm(city, network_type=network_type)
    streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_nodes=True)
    
    buildings = data_input.load_buildings_from_osm(city)
    buildings = pp.get_buildings(buildings=buildings, streets=streets, intersections=junctions, local_crs=local_crs)
    
    # Generate tessellations
    tessellations, enclosures = pp.get_tessellation(buildings=buildings, streets=streets, 
                                                    tess_mode='enclosed', clim='adaptive')
    
    # Generate metrics
    t0 = time()
    print(f'Generating building metrics for {city}')
    buildings = metrics.generate_building_metrics(buildings)
    print(f'Building metrics: {(t1:=time())-t0:.2f} s')
    
    print(f'Generating graph-related building metrics for {city}')
    buildings, streets = metrics.generate_graph_metrics(buildings, streets, tessellations)
    print(f'Graph-related building metrics: {time()-t1:.2f} s')

    metrics.generate_streets_metrics(streets)
    junctions, streets = metrics.generate_junctions_metrics(streets)
    
    # Save data
    buildings.to_file(f'{output_folder}/{city}_buildings.geojson', driver='GeoJSON')
    streets.to_file(f'{output_folder}/{city}_streets.geojson', driver='GeoJSON')
    junctions.to_file(f'{output_folder}/{city}_junctions.geojson', driver='GeoJSON')
    tessellations.to_file(f'{output_folder}/{city}_tessellations.geojson', driver='GeoJSON')
    
    print(f'Data saved for {city}')

def join_dfs(cities, config_file='config.ini'):
    # Read or create configuration file
    config = configparser.ConfigParser()
    
    if not os.path.isfile(config_file):
        config['Paths'] = {
            'gdb_folder': '/path/to/your/gdb/folder',
            'output_dir': '/path/to/your/output/folder',
            'output_folder': '/path/to/your/output/folder'
        }
        with open(config_file, 'w') as configfile:
            config.write(configfile)
        print(f"Created configuration file '{config_file}'.")
    
    config.read(config_file)
    output_folder = config['Paths']['output_folder']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize an empty DataFrame to store metrics for each city
    metrics_df = pd.DataFrame()

    for city in cities:
        city_metrics = pd.DataFrame()
        try:
            # Load the saved GeoJSON for buildings
            buildings = gpd.read_file(os.path.join(output_folder, f'{city}_buildings.geojson'))
            streets = gpd.read_file(os.path.join(output_folder, f'{city}_streets.geojson'))
            junctions = gpd.read_file(os.path.join(output_folder, f'{city}_junctions.geojson'))

            # Get all numeric columns (metrics) for buildings
            building_numeric_columns = buildings.select_dtypes(include=['number']).columns

            # Calculate mean for each building numeric column
            building_metrics = buildings[building_numeric_columns].mean().to_frame().T
            building_metrics.columns = [f'building_{col}' for col in building_metrics.columns]  # Prefix with 'building_'
            # Add to city_metrics
            city_metrics = pd.concat([city_metrics, building_metrics], axis=1)

            # Get all numeric columns (metrics) for streets
            street_numeric_columns = streets.select_dtypes(include=['number']).columns

            # Calculate mean for each street numeric column
            street_metrics = streets[street_numeric_columns].mean().to_frame().T
            street_metrics.columns = [f'street_{col}' for col in street_metrics.columns]  # Prefix with 'street_'

            # Add to city_metrics
            city_metrics = pd.concat([city_metrics, street_metrics], axis=1)

            # Get all numeric columns (metrics) for junctions
            junction_numeric_columns = junctions.select_dtypes(include=['number']).columns

            # Calculate mean for each junction numeric column
            junction_metrics = junctions[junction_numeric_columns].mean().to_frame().T
            junction_metrics.columns = [f'junction_{col}' for col in junction_metrics.columns]  # Prefix with 'junction_'

            # Add to city_metrics
            city_metrics = pd.concat([city_metrics, junction_metrics], axis=1)

            city_metrics['City'] = city  # Add city name
            metrics_df = pd.concat([metrics_df, city_metrics], ignore_index=True)

        except FileNotFoundError:
            print(f'GeoJSON file for {city} not found, skipping...')
        except KeyError:
            print(f'Error processing {city}, skipping...')

    # Reorder columns so that 'City' is the first column
    cols = ['City'] + [col for col in metrics_df.columns if col != 'City']
    metrics_df = metrics_df[cols]

    # Save the metrics DataFrame as a CSV
    metrics_df.to_csv(os.path.join(output_folder, 'city_metrics.csv'), index=False)

    print(metrics_df)

# Loop through all cities
for city in cities:
    process_city(city)
    join_dfs(cities)