import geopandas as gpd
import momepy
import pandas

def get_buildings(buildings, local_crs=None):
    ngeoms = len(buildings.geom_type.value_counts())
    if ngeoms > 1:
        if sum(buildings.geom_type == 'Polygon') > 0:
            polygontype = 'Polygon'
        elif sum(buildings.geom_type == 'MultiPolygon') > 0:
            polygontype = 'MultiPolygon'
        else:
            raise ValueError('No polygons or multi-polygons in building data')
        buildings = buildings[buildings.geom_type==polygontype].reset_index(drop=True)
        if local_crs:
            buildings.to_crs(local_crs)
    
    # This is not perfect. It assumes that in case there's more than one height value, the building height is the 1st.
    if 'height' not in buildings.keys():
        if len(height_vars := buildings.keys()[['height' in key for key in buildings.keys()]]) == 0:
            print('No building heights in data')
            return buildings[['geometry']]
        elif len(height_vars) == 1:
            buildings['height'] = buildings[height_vars]
        else:
            print('here')
            buildings['height'] = buildings[height_vars[0]]
        
    return buildings[['geometry', 'height']]

def get_streets(streets):
    streets = momepy.remove_false_nodes(streets)
    streets = streets[['geometry']]
    streets['length'] = streets.length
    
    return streets

if __name__=='__main__':
    import os
    from sys import path
    from pathlib import Path    
    if not (pardir := str(Path(__file__).absolute().parent)) in path:
        path.append(pardir)
    import osmnx
    
    place = 'Jerusalem'
    local_crs = 'EPSG:2039'
    network_type = 'drive'

    # datadir = '../../Michaels_data/All_layers_2'
    # gdb_file = os.path.join(datadir, os.listdir(datadir)[76])
    # print(gdb_file)
    # gdf = gpd.read_file(gdb_file)
    # buildings = get_buildings(gdf)
    # print(gdf.head())
    # print(buildings.head())

    osm_graph = osmnx.graph_from_place(place, network_type=network_type)
    osm_graph = osmnx.projection.project_graph(osm_graph, to_crs=local_crs)
    osm_streets = osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(osm_graph),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    ).reset_index(drop=True)
    print(osm_streets.head())

    streets = get_streets(osm_streets)
    print(streets.head())

