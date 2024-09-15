import geopandas as gpd
import momepy
import pandas
import numpy as np
from time import time

def get_buildings(buildings, local_crs=None, height_name=None, min_area=20):
    '''
    get_buildings(buildings, local_crs, height_name, min_area)
    input:
        - buildings : dtaframe containing the relevant buildings
        - local_crs (optional) : coordinate reference system into which to transform the buildings.
        - height_name (optional) : name of building height variable. If not availbalbe the code looks
          for variables containing "height" in their name, and chose the 1st of them.
        - min_area (default 20) : minimal area of building to take into account. Defaults to 20 m^2
    output:
        - minimal_buildings : dataframe of the buildings containing only geometries and height (if available)
    
    Function outputs buildings as polygons. Multipolygons are exploded to multiple polygons.
    Removes duplicated buildings, by geometry.
    Does not (yet) remove overlapping but not duplicated buildings!
    '''
    buildings = buildings.to_crs(local_crs) if local_crs is not None else buildings
    categories = ['geometry']
    if height_name:
        buildings['height'] = buildings[height_name]
        categories.append('height')
    else:
        if 'height' not in buildings.keys():
            if len(height_vars := buildings.keys()[['height' in key for key in buildings.keys()]]) == 0:
                print('No building heights in data')
            elif len(height_vars) == 1:
                buildings['height'] = buildings[height_vars]
                categories.append('height')
            else:
                buildings['height'] = buildings[height_vars[0]]
                categories.append('height')
    buildings = buildings[categories]
    buildings = buildings.explode(index_parts=False).reset_index(drop=True)
    buildings.geometry = buildings.buffer(0)
    buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)
    buildings = buildings.drop_duplicates(['geometry']).reset_index(drop=True)
    buildings = buildings[buildings.area > min_area]


    return buildings.reset_index(drop=True)

def get_streets(streets, local_crs=None):
    '''
    get_streets(streets, local_crs)
    input:
        - streets : dtaframe containing the relevant streets
        - local_crs (optional) : coordinate reference system into which to transform the buildings.
    output:
        - minimal_streets : dataframe of the streets containing only geometries and length of streets
    
    Function does not check for geometry type. LineString assumed, PolyLineString should also work propertly.
    Closes gaps between street segments and removes false nodes
    Removes duplicated streets, if any present.
    Does not, yet, consolidate multiple crossings at intersection into single node.
    '''

    streets.to_crs(local_crs) if local_crs else streets
    streets = momepy.remove_false_nodes(streets)
    streets.geometry = momepy.close_gaps(streets, tolerance=0.25)
    streets = momepy.roundabout_simplification(streets)

    # Find way to consolidate networks - something wrong with streets as graph...
    # streets = momepy.consolidate_intersections(streets, tolerance=30)
    streets = streets.drop_duplicates('geometry').reset_index(drop=True)

    streets['length'] = streets.length
    streets = streets[['geometry', 'length']]

    return streets

def tessellation(buildings, streets=None, tess_mode='enclosed', clim='adaptive'):
    '''
    TBD
    Creates tessellations from buildings dataframe, if "enclosed" mode then uses streets as enclosures.
    '''
    try:
        limit = momepy.buffered_limit(buildings, clim)
    except:
        if clim=='adaptive':
            print('Failed creating limit in adaptive mode. Resetting limit to 100')
            clim = 100
            limit = momepy.buffered_limit(buildings, clim)
    if tess_mode=='enclosed':
        if streets is None:
            raise ValueError('No streets provided for enclosed tessellation')
        t0 = time()
        enclosures = momepy.enclosures(streets, limit)
        enc_time = time()
        try:
            print('Trying enclosed tessellations')
            tessellations = momepy.enclosed_tessellation(buildings, enclosures)
        except:
            try:
                print('Failed producing enclosed tessellations')
                print('Trying morphometric tessellations')
                tessellations, limit = tessellation(buildings, tess_mode='morphometric')
            except:
                if clim=='adaptive':
                    print('Faild with "adaptive" limits. Resetting limit to 100 and retrying morphological tessellation')
                    tessellations, limit = tessellation(buildings, tess_mode='morphometric', clim=100)

        tess_time = time()
    
        print(f'Computed enclosures: {enc_time-t0} sec\nComputed tessellations: {tess_time-enc_time} sec')

        return tessellations, enclosures, limit
    elif tess_mode=='morphometric':
        t0 = time()
        tessellations = momepy.morphological_tessellation(buildings, clip=limit)
        print(f'Computed tessellations: {time()-t0} s')
        return tessellations, limit
    
    else:
        raise NotImplementedError('Requested tessellation mode not implemented')
        
def find_overlaps(gdf):
    # Buffer
    buffer = gdf.copy()
    buffer.geometry = buffer.geometry.buffer(3)

    # Overlaps check
    intersecting_gdf = buffer.sjoin(buffer, predicate="intersects")
    intersecting_gdf = intersecting_gdf.loc[
        intersecting_gdf.index != intersecting_gdf.index_right
    ]
    overlaps_ids = np.unique(intersecting_gdf.index)  # list overlapping IDs
    return(overlaps_ids)


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
    datadir = '/Users/amirkl/PhD/DSSG/michaels_data/All_layers'
    gdb_file = Path(datadir, os.listdir(datadir)[-1], 'commondata', 'jps_reka.gdb')
    # print(gdb_file)
    gdf = gpd.read_file(gdb_file)
    ### Add check of geometry type to OSM datasets?
    # gdf = osmnx.features_from_place(place, tags={'building':True})
    buildings = get_buildings(gdf, local_crs='EPSG:2039')
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
    # print(len(osm_streets))
    # print(osm_streets.head())

    streets = get_streets(osm_streets)
    # print(len(streets))
    # print(streets.head())
    overlaps = find_overlaps(buildings)
    tessellations, enclosures, limits = tessellation(buildings, streets, tess_mode='enclosed' )
    # tessellations, limits = tessellation(buildings, tess_mode='morphometric' )

