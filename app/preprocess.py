import geopandas as gpd
import momepy
import pandas

def get_buildings(buildings, local_crs=None):
    ngeoms = len(buildings.geom_type.value_counts())
    if ngeoms > 1:
        if sum(buildings.building_type == 'Polygon'):
            polygontype = 'Polygon'
        elif sum(buildings.build_type == 'MultiPolygon'):
            polygontype = 'MultiPolygon'
        else:
            raise ValueError('No polygons or multi-polygons in building data')
        buildings = buildings[buildings.geom_type=='Polygon'].reset_index(drop=True)[['geometry']]
        if local_crs:
            buildings.to_crs(local_crs)
    return buildings

if __name__=='__main__':
    import os
    from sys import path
    from pathlib import Path    
    if not (pardir := str(Path(__file__).absolute().parent)) in path:
        path.append(pardir)
    
    datadir = '../../../Michaels_data/All_layers_2'
    gdb_file = os.path.join(datadir, os.listdir(datadir)[76])
    print(gdb_file)
    gdf = gpd.read_file(gdb_file)


