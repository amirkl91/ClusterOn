from time import time
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm, load_shapefile
import metrics
import merge_dfs as md
from data_input import load_gdb_layer
import config_parser
from generate_clusters import get_cgram, add_cluster_col, plot_clusters
# from data_output import dataframe_to_gdb
import stats_generation2 as statgen


verbose=True
t0 = time()
place = 'Jerusalem'
local_crs = 'EPSG:2039'

# buildings data (Michael)
# configpath = '/Users/amirkl/PhD/DSSG/DSSG_mapping_Jerus/config.ini'
# params = config_parser.read_config(configpath)
# gdb_bld_path = '/Users/amirkl/PhD/DSSG/michaels_data/shp/棄墠.shp'
# buildings = load_shapefile(gdb_bld_path)
# buildings = load_gdb_layer(gdb_bld_path)
buildings = load_buildings_from_osm(place)


# streets data (OSM)
network_type = 'drive'
streets = load_roads_from_osm(place, network_type=network_type)

#Preprocess
streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_juncions=True)
buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs, )

tessellations = pp.get_tessellation(buildings=buildings, streets=streets, 
                                        tess_mode='morphometric', clim='adaptive')

### Get metrics
metrics.generate_building_metrics(buildings)
queen_1 = metrics.generate_tessellation_metrics(tessellations, buildings)
metrics.generate_streets_metrics(streets)
queen_3 = metrics.generate_graph_building_metrics(buildings, streets, queen_1)
junctions, streets = metrics.generate_junctions_metrics(streets)

### Merge dataframes
merged = md.merge_all_metrics(tessellations, buildings, streets, junctions)
metrics_with_percentiles = md.compute_percentiles(merged, queen_3)
standardized = md.standardize_df(metrics_with_percentiles)

from helper_functions import select_best_num_of_clusters

best_scores = select_best_num_of_clusters(standardized, standardize=False, min_cluster=1, max_cluster=20, n_init=30, random_state=None, plot=True)

'''
### make clusters
cgram = get_cgram(standardized, 14)
urban_types = add_cluster_col(merged, buildings, cgram, 10)
plot_clusters(urban_types)

# gdf.drop(columns="geometry").to_csv("output.csv", index=False)
threshold = len(urban_types) * 0.5

# Step 1: Remove columns with more than 50% NaNs
gdf = urban_types.dropna(axis=1, thresh=threshold)

# Step 2: Replace remaining NaNs with 0 in the remaining columns
gdf.fillna(0, inplace=True)
gdf.drop(
    columns=["street_index", "junction_index"],
    inplace=True,
)

statgen.analyze_gdf(gdf, "cluster", "../../output_CSVs")
'''