import geopandas as gpd
import fiona
import os
import osmnx

def load_gdb_layer_from_folder(gdb_folder, gdb_index=0, layer_index=0):
    """
    Load a specific layer from a .gdb file in the given folder.

    Parameters:
    - gdb_folder: str, path to the folder containing .gdb files
    - gdb_index: int, index of the .gdb file to select
    - layer_index: int, index of the layer within the selected .gdb file

    Returns:
    - gdf: GeoDataFrame, the loaded data
    """
    # List all .gdb directories in the folder
    gdb_files = [os.path.join(gdb_folder, f) for f in os.listdir(gdb_folder) if f.endswith('.gdb')]
    if not gdb_files:
        raise ValueError("No .gdb files found in the specified folder.")

    print("GDB Files:", gdb_files)

    if gdb_index >= len(gdb_files):
        raise IndexError("Selected .gdb index is out of range.")

    # Choose the specific .gdb file
    gdb_file = gdb_files[gdb_index]

    # List the layers in the selected .gdb
    layers = fiona.listlayers(gdb_file)
    if not layers:
        raise ValueError("No layers found in the selected .gdb file.")

    print("Layers in the selected GDB:", layers)

    if layer_index >= len(layers):
        raise IndexError("Selected layer index is out of range.")

    # Choose the specific layer within the .gdb
    layer_name = layers[layer_index]

    # Load the specific layer
    gdf = gpd.read_file(gdb_file, layer=layer_name)
    return gdf

def load_gdb_layer(gdb_file, layer_index=0):
    """
    Load a specific layer from a .gdb file.

    Parameters:
    - gdb_file: str, path to the .gdb file
    - layer_index: int, index of the layer within the .gdb file

    Returns:
    - gdf: GeoDataFrame, the loaded data
    """
    # List the layers in the selected .gdb
    layers = fiona.listlayers(gdb_file)
    if not layers:
        raise ValueError("No layers found in the specified .gdb file.")
    
    print("Layers in the selected GDB:", layers)

    if layer_index >= len(layers):
        raise IndexError("Selected layer index is out of range.")

    # Choose the specific layer within the .gdb
    layer_name = layers[layer_index]

    # Load the specific layer
    gdf = gpd.read_file(gdb_file, layer=layer_name)
    return gdf

def load_shapefile(shapefile_path):
    """
    Load a shapefile (.shp) from the given path.

    Parameters:
    - shapefile_path: str, path to the .shp file

    Returns:
    - gdf: GeoDataFrame, the loaded data
    """
    # Check if the shapefile exists
    if not os.path.isfile(shapefile_path):
        raise FileNotFoundError(f"The shapefile '{shapefile_path}' does not exist.")
    
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    return gdf

def load_buildings_from_osm(place):
    buildings = osmnx.features_from_place(place, tags={'building':True})
    buildings = buildings[buildings.geom_type=='Polygon'].reset_index(drop=True)
    return buildings

def load_roads_from_osm(place, network_type):
    osm_graph = osmnx.graph_from_place(place, network_type='drive')
    osm_graph = osmnx.projection.project_graph(osm_graph)
    streets = osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(osm_graph),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    ).reset_index(drop=True)
    return streets



# Example usage
if __name__ == "__main__":
    # Define the folder containing .gdb files
    gdb_folder = '/Users/annarubtsov/Desktop/layers'
    # define gdb file path
    gdb_file = '/Users/annarubtsov/Desktop/myproject16.gdb'
    shapefile_path = '/Users/annarubtsov/Desktop/shp_name/shp_name.shp'
    try:
        gdf = load_gdb_layer_from_folder(gdb_folder=gdb_folder, gdb_index=0, layer_index=0)
        print(gdf.head())

        # gdf = load_gdb_layer(gdb_file=gdb_file, layer_index=0)
        # print(gdf.head())

        # gdf_shapefile = load_shapefile(shapefile_path=shapefile_path)
        # print("Data from shapefile:")
        # print(gdf_shapefile.head())
    except Exception as e:
        print(f"An error occurred: {e}")