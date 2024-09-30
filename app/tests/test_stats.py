import sys
sys.path.append('../')
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
junctions, streets = metrics.generate_junctions_metrics(streets)
buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs, )

tessellations = pp.get_tessellation(buildings=buildings, streets=streets, 
                                        tess_mode='morphometric', clim='adaptive')

### Get metrics
metrics.generate_building_metrics(buildings)
queen_1 = metrics.generate_tessellation_metrics(tessellations, buildings)
metrics.generate_streets_metrics(streets)
queen_3 = metrics.generate_graph_building_metrics(buildings, streets, queen_1)

### Merge dataframes
merged = md.merge_all_metrics(tessellations, buildings, streets, junctions)
metrics_with_percentiles = md.compute_percentiles(merged, queen_3)
standardized = md.standardize_df(metrics_with_percentiles)

import generate_clusters as gc
fig, ax = gc.plot_num_of_clusters(standardized, standardize=False, min_clusters=2, max_clusters=20, random_state=None)
fig.show()
best_scores = gc.best_davies_bouldin_score(standardized, standardize=False, min_clusters=2, max_clusters=20, random_state=None)


import contextily as ctx
import matplotlib.pyplot as plt

cgram = gc.get_cgram(standardized, 20)

urban_types = gc.add_cluster_col(merged, buildings, cgram, 10)
import matplotlib.patches as mpatches
cmap='tab20'
colors = plt.get_cmap(cmap).colors
categories = merged['cluster'].unique()
legend_handles = [mpatches.Patch(color=colors[i], label=f'Cluster {category+1:02d}') 
                    for i, category in enumerate(categories)]
key = lambda patch: patch.get_label()
sorted_patches = sorted(legend_handles, key=key)

fig, ax = plt.subplots(figsize=(8, 12))
# cax = fig.add_axes([0.25,0.35,0.03,0.4])
urban_types.plot(column='cluster', cmap=cmap, legend=False, ax=ax,)
ctx.add_basemap(ax, crs=streets.crs, source=ctx.providers.CartoDB.Positron)
ax.set_axis_off()
ax.legend(handles=sorted_patches, title='Cluster', loc='upper left')
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