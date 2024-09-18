from time import time
from app import preprocess as pp
from app import data_input
from app import metrics
import pandas as pd
import geopandas as gpd
import os
import re
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Cities and their CRS
city_names = ['Jerusalem', 'Tel Aviv', 'Haifa', 'Rishon LeZion', 'Petah Tikva', 'Ashdod',
           'Netanya', 'Beer Sheva', 'Bnei Brak', 'Holon', 'Ramat Gan', 'Rehovot', 'Ashkelon',
           'Bat Yam', 'Beit Shemesh', 'Kfar Saba', 'Herzliya', 'Hadera', 'Modiin', 'Nazareth']

crs_mapping = {city: "EPSG:2039" for city in city_names}  # Mapping CRS to cities

hebrew_to_english = {
    'ירושלים': 'Jerusalem',
    'תל אביב-יפו': 'Tel Aviv',
    'חיפה': 'Haifa',
    'ראשון לציון': 'Rishon LeZion',
    'פתח תקווה': 'Petah Tikva',
    'אשדוד': 'Ashdod',
    'נתניה': 'Netanya',
    'באר שבע': 'Beer Sheva',
    'בני ברק': 'Bnei Brak',
    'חולון': 'Holon',
    'רמת גן': 'Ramat Gan',
    'רחובות': 'Rehovot',
    'אשקלון': 'Ashkelon',
    'בת ים': 'Bat Yam',
    'בית שמש': 'Beit Shemesh',
    'כפר סבא': 'Kfar Saba',
    'הרצליה': 'Herzliya',
    'חדרה': 'Hadera',
    'מודיעין-מכבים-רעות': 'Modiin',
    'נצרת': 'Nazareth'
}

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
    output_dir = config['Paths']['output_dir']

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

def calculate_errors(df):
    """
    Calculate standard error of the mean (SEM) for each numeric column in the DataFrame.
    
    Parameters:
    - df: DataFrame containing numeric columns.
    
    Returns:
    - errors_df: DataFrame containing SEM for each numeric column.
    """
    errors = {}
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            # Calculate SEM
            sem = df[column].sem()
            errors[column] = sem
    errors_df = pd.DataFrame(errors, index=['SEM']).T
    return errors_df

def join_dfs(city_names, config_file='config.ini'):
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

    # Initialize an empty DataFrame to store metrics for each city
    metrics_df = pd.DataFrame()
    sem_df = pd.DataFrame()  # This will store the SEM values

    for city in city_names:
        city_metrics = pd.DataFrame()
        city_sem = pd.DataFrame()
        try:
            # Load the saved GeoJSON for buildings
            buildings = gpd.read_file(os.path.join(output_folder, f'{city}_buildings.geojson'))
            streets = gpd.read_file(os.path.join(output_folder, f'{city}_streets.geojson'))
            junctions = gpd.read_file(os.path.join(output_folder, f'{city}_junctions.geojson'))

            # Get all numeric columns (metrics) for buildings
            building_numeric_columns = buildings.select_dtypes(include=['number']).columns

            # Calculate mean for each building numeric column
            building_metrics_mean = buildings[building_numeric_columns].mean().to_frame().T
            building_metrics_mean.columns = [f'building_{col}' for col in building_metrics_mean.columns]  # Prefix with 'building_'
            # Calculate SEM for each building numeric column
            building_metrics_sem = buildings[building_numeric_columns].sem().to_frame().T
            building_metrics_sem.columns = [f'building_{col}_sem' for col in building_metrics_sem.columns]  # Prefix with 'building_' and add '_sem'

            # Add to city_metrics
            city_metrics = pd.concat([city_metrics, building_metrics_mean], axis=1)
            city_sem = pd.concat([city_sem, building_metrics_sem], axis=1)

            # Get all numeric columns (metrics) for streets
            street_numeric_columns = streets.select_dtypes(include=['number']).columns

            # Calculate mean for each street numeric column
            street_metrics_mean = streets[street_numeric_columns].mean().to_frame().T
            street_metrics_mean.columns = [f'street_{col}' for col in street_metrics_mean.columns]  # Prefix with 'street_'
            # Calculate SEM for each street numeric column
            street_metrics_sem = streets[street_numeric_columns].sem().to_frame().T
            street_metrics_sem.columns = [f'street_{col}_sem' for col in street_metrics_sem.columns]  # Prefix with 'street_' and add '_sem'
            
            # Add to city_metrics
            city_metrics = pd.concat([city_metrics, street_metrics_mean], axis=1)
            city_sem = pd.concat([city_sem, street_metrics_sem], axis=1)
            
            # Get all numeric columns (metrics) for junctions
            junction_numeric_columns = junctions.select_dtypes(include=['number']).columns

            # Calculate mean for each junction numeric column
            junction_metrics_mean = junctions[junction_numeric_columns].mean().to_frame().T
            junction_metrics_mean.columns = [f'junction_{col}' for col in junction_metrics_mean.columns]  # Prefix with 'junction_'
            # Calculate SEM for each junction numeric column
            junction_metrics_sem = junctions[junction_numeric_columns].sem().to_frame().T
            junction_metrics_sem.columns = [f'junction_{col}_sem' for col in junction_metrics_sem.columns]  # Prefix with 'junction_' and add '_sem'
            
            # Add to city_metrics and city_sem
            city_metrics = pd.concat([city_metrics, junction_metrics_mean], axis=1)
            city_sem = pd.concat([city_sem, junction_metrics_sem], axis=1)

            # Add the city name to both DataFrames
            city_metrics['city'] = city
            city_sem['city'] = city
            
            # Append metrics and SEM to the main DataFrames
            metrics_df = pd.concat([metrics_df, city_metrics], ignore_index=True)
            sem_df = pd.concat([sem_df, city_sem], ignore_index=True)

        except FileNotFoundError:
            print(f'GeoJSON file for {city} not found, skipping...')
        except KeyError:
            print(f'Error processing {city}, skipping...')

    # Reorder columns so that 'City' is the first column
    cols_metrics = ['city'] + [col for col in metrics_df.columns if col != 'city']
    cols_sem = ['city'] + [col for col in sem_df.columns if col != 'city']
    metrics_df = metrics_df[cols_metrics]
    sem_df = sem_df[cols_sem]

    # Save the metrics DataFrame as a CSV
    metrics_df.to_csv(os.path.join(output_folder, 'city_metrics.csv'), index=False)
    sem_df.to_csv(os.path.join(output_folder, 'city_sem.csv'), index=False)

    print(metrics_df)

# Function to translate city names
def translate_city(name):
    return hebrew_to_english.get(name, name)  # If not found, return the original name

def city_properties_df(config_file='config.ini'):# Read or create configuration file
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
    flatness_csv_path = config['Paths']['flatness_csv_path']

    # Read the flatness data from the CSV
    flatness_df = pd.read_csv('data/flatness.csv')
    
    # add the cities database from Wikipedia
    # Source URL: https://he.wikipedia.org/wiki/ערים_בישראל
    # Then use this tool to download:
    #  https://wikitable2csv.ggor.de
    # I then manually translated the column names and saved as CSV
    # Use table selector: .toccolours

    cities = pd.read_csv('data/cities_israel.csv')
    # dictionary of Hebrew to English city names

    # Apply the translation to the 'name' column
    cities['name_english'] = cities['name'].apply(translate_city)

    # remove various signs
    cities_columns = ['name_english', 'area', 'density', 'population', 'growth rate', 'socioeconomic', 'year founded']

    for c in cities_columns:
        if c in ['name', 'name_english']:
            continue
        cities[c] = cities[c].astype(str).apply(lambda x: re.sub(r'[%,]', '', x)).astype(float)        

    cities = cities[cities_columns]
    cities.rename(columns={'name_english': 'city'}, inplace=True)
    cities = cities[cities['city'].isin(city_names)]
    
    # Merge the cities_df with flatness_df on the 'City' column
    cities = pd.merge(cities, flatness_df, on='city', how='left')
    cities.drop(columns=['num_sampled_points'])

    cities.to_csv(os.path.join(output_folder, 'city_properties.csv'), index=False)
    return cities

def find_corr(config_file='config.ini'):
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

    # Construct the full path to city_metrics.csv in the output folder
    city_metrics_path = os.path.join(output_folder, 'city_metrics.csv')
    city_properties_path = os.path.join(output_folder, 'city_properties.csv')

    # Read the CSV files into DataFrames
    city_metrics = pd.read_csv(city_metrics_path)
    city_properties = pd.read_csv(city_properties_path)

    columns_to_filter = [
    col for col in city_metrics.columns
    if any(substring in col for substring in ['_id', 'ID', '_index', 'end', 'start'])]
    city_metrics = city_metrics.drop(columns=columns_to_filter)

    # Apply log transformation to 'density' and 'population'
    city_properties['density'] = np.log1p(city_properties['density'])
    city_properties['population'] = np.log1p(city_properties['population'])

    # Assuming both DataFrames have a 'city' column and you want to merge on this column
    merged_df = pd.merge(city_metrics, city_properties, on='city')

    # Separate the city_metrics and city_properties columns after merging
    metrics_columns = city_metrics.columns.difference(['city'])
    properties_columns = city_properties.columns.difference(['city'])

    # Calculate the correlation between the columns of city_metrics and city_properties
    correlations = {}
    for metric_col in metrics_columns:
        for property_col in properties_columns:
            correlation = merged_df[metric_col].corr(merged_df[property_col], method="spearman")
            correlations[(metric_col, property_col)] = correlation

    # Convert the result to a DataFrame for easy viewing
    correlation_df = pd.DataFrame(list(correlations.items()), columns=['Metric_Property_Pair', 'Correlation_Value'])

    # Sort the DataFrame by correlation values
    correlation_df = correlation_df.sort_values(by='Correlation_Value', key=lambda x: x.abs(), ascending=False)

    # Save the metrics DataFrame as a CSV
    correlation_df.to_csv(os.path.join(output_folder, 'correlation.csv'), index=False)

    print(correlation_df)

def plot_features(city_metrics, city_properties, x_err_values, col1, col2):
    """
    Plot the relationship between two features, one from city_metrics and one from city_properties.
    
    Parameters:
    - city_metrics: DataFrame containing city metrics.
    - city_properties: DataFrame containing city properties.
    - col1: Feature from city_metrics to plot on the x-axis.
    - col2: Feature from city_properties to plot on the y-axis.
    """

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

    # Construct the full path to city_metrics.csv in the output folder
    city_metrics_path = os.path.join(output_folder, 'city_metrics.csv')
    city_properties_path = os.path.join(output_folder, 'city_properties.csv')


    # Merge the two DataFrames on the 'city' column, along with the x_err_values (error bars)
    merged_df = pd.merge(city_metrics, city_properties, on='city')
    merged_df = pd.merge(merged_df, x_err_values[['city', col1+'_sem']], on='city')  # Assuming col1_sem is the error column

    # Create a scatter plot using matplotlib
    plt.figure(figsize=(10, 6))
    
    # Get unique cities
    unique_cities = merged_df['city'].unique()
    
    # Generate a color map
    colors = cm.get_cmap('tab20', len(unique_cities))

    # Scatter plot with color map and error bars
    for idx, city in enumerate(unique_cities):
        subset = merged_df[merged_df['city'] == city]
        
        # Plot each city with error bars
        plt.errorbar(subset[col1], subset[col2], xerr=subset[col1 + '_sem'], fmt='o', label=city, color=colors(idx), capsize=5)

        # Annotate each point with the city name
        for i, row in subset.iterrows():
            plt.text(row[col1], row[col2], row['city'], fontsize=8, ha='right')

    # Adding labels and title
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Scatter Plot of {col1} vs {col2} by City (with Error Bars)')
    
    # Adding legend
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show plot
    plt.tight_layout()
    plt.show()

############################ RUN: #################################

# for city in city_names:
#     process_city(city)

#join_dfs(city_names)
#city_properties_df()


#find_corr()

config = configparser.ConfigParser()
config_file = 'config.ini'
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

# Read the CSV files into DataFrames
city_metrics = pd.read_csv(os.path.join(output_folder, 'city_metrics.csv'))
city_properties = pd.read_csv(os.path.join(output_folder, 'city_properties.csv'))
city_sem = pd.read_csv(os.path.join(output_folder, 'city_sem.csv'))

correlation_df = pd.read_csv(os.path.join(output_folder, 'correlation.csv'))
# Get the first 10 metric-property pairs
top_10_pairs = correlation_df.head(10)
# Loop through the first 10 metric-property pairs and plot each
for index, row in top_10_pairs.iterrows():
    # Extract metric and property from the tuple string
    metric, property = eval(row['Metric_Property_Pair'])
    # Get the corresponding SEM for the metric
    metric_sem = f"{metric}_sem"
    # Plot features
    plot_features(city_metrics, city_properties, city_sem[['city', metric_sem]], metric, property)

