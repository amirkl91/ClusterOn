from time import time
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm
import metrics
import merge_dfs as md
import geopandas as gpd

verbose = True
t0 = time()
place = "modiin"
local_crs = "EPSG:2039"
network_type = "drive"

### Get dataframes
streets = load_roads_from_osm(place, network_type=network_type)
streets, junctions = pp.get_streets(
    streets=streets, local_crs=local_crs, get_juncions=True
)
buildings = load_buildings_from_osm(place)
buildings = pp.get_buildings(
    buildings=buildings,
    streets=streets,
    junctions=junctions,
    local_crs=local_crs,
)
tessellations = pp.get_tessellation(
    buildings=buildings, streets=streets, tess_mode="morphometric", clim="adaptive"
)

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
print("geometry" in merged.columns)
print("number of columns:", len(merged.columns))
# david
# Check if standardized DataFrame has a geometry column
geometry_column = merged["geometry"]

# Step 2: Merge the geometry column into the standardized DataFrame
standardized_with_geometry = standardized.copy()
standardized_with_geometry["geometry"] = geometry_column

# Step 3: Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(standardized_with_geometry, geometry="geometry", crs=local_crs)

# Now you can save it to a file
merged = md.cluster_gdf(merged)

merged.to_file("../gpkg_files/modiin.gpkg", layer="nodes", driver="GPKG")
