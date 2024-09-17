import os
import fiona
import geopandas as gpd
import preprocess as pp
from data_input import load_buildings_from_osm, load_roads_from_osm

place = 'Jerusalem'
local_crs = 'EPSG:2039'
network_type = 'drive'

streets = load_roads_from_osm(place, network_type=network_type)
streets, junctions = pp.get_streets(streets=streets, local_crs=local_crs, get_nodes=True)

# buildings = load_buildings_from_osm(place)
# buildings = pp.get_buildings(buildings=buildings, streets=streets, junctions=junctions, local_crs=local_crs, )

# tesselations, enclosures = pp.get_tessellation(buildings=buildings, streets=streets, 
#                                             tess_mode='enclosed', clim='adaptive')

root_folder = '../../michaels_data/All_layers'
gdb_files = []
# List all .gdb files in the "commondata" folder
for dirpath, dirnames, filenames in os.walk(root_folder):
    # Check if the folder contains a "commondata" folder
    if "commondata" in dirnames:
        commondata_folder = os.path.join(dirpath, "commondata")
        commondata_folder = os.path.normpath(commondata_folder)
        for f in os.listdir(commondata_folder):
            if f.endswith('.gdb'):
                gdb_files.append(os.path.join(commondata_folder, f))

# Print the collected .gdb files
# for idx, gdb in enumerate(gdb_files):
#     print(f'{idx} {gdb}')
# print("GDB Files:", gdb_files)
# load municipal buildings
gdb_file = gdb_files[6]  # Modify if you want to choose a different .gdb
print(f'Chosen gdb file: {gdb_file}')
# List the layers in the selected .gdb
layers = fiona.listlayers(gdb_file)
# print("Layers in the selected GDB:", layers)

# Choose a specific layer within the .gdb
textures_layer = layers[0]  # Modify if you want to choose a different layer

# Load the specific layer
gdf = gpd.read_file(gdb_file, layer=textures_layer)
buildings = pp.get_buildings(gdf, streets, junctions)
# buildings = gdf[["geometry"]]
# m_buildings = m_buildings.to_crs(osm_buildings.crs)

tesselations, enclosures = pp.get_tessellation(buildings=buildings, streets=streets, 
                                            tess_mode='enclosed', clim='adaptive')
