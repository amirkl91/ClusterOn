from time import time
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm
import metrics
import merge_dfs as md
from data_input import load_gdb_layer
import config_parser
from generate_clusters import get_cgram, add_cluster_col, plot_clusters
from data_output import dataframe_to_gdb

# buildings data (Michael)
params = config_parser.read_config('config.ini')
gdb_bld_path = params['gdb_bld_path']
buildings = load_gdb_layer(gdb_bld_path)

# streets data (OSM)
verbose=True
t0 = time()
place = 'Jerusalem'
local_crs = 'EPSG:2039'
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

### make clusters
cgram = get_cgram(standardized, 14)
urban_types = add_cluster_col(merged, buildings, cgram, 13)
plot_clusters(urban_types)
dataframe_to_gdb(urban_types, "/Users/annarubtsov/Desktop/DSSG/Michaels_Data/All_Layers/מרקמים/commondata/myproject16.gdb", "urban_types")