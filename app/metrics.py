import geopandas as gpd
import momepy

from libpysal import graph

def generate_building_metrics(buildings: gpd.geodataframe, height_column_name=None, selected=None):
    buildings['Area'] = buildings.geometry.area

    if 'height' in buildings.columns:
        buildings['floor_area'] = momepy.floor_area(buildings['Area'], buildings['building_height'])
        buildings['volume'] = momepy.volume(buildings['Area'], buildings['building_height'])
        buildings['form_factor'] = momepy.form_factor(buildings, buildings['building_height'])

    # Check if all geometries are either Polygon or MultiPolygon
    if (buildings.geometry.geom_type == 'Polygon').all():
        buildings['corners'] = momepy.corners(buildings)
        if selected is None or selected['squareness']:
            buildings['squareness'] = momepy.squareness(buildings)
        # buildings_centroid_corner_distance = momepy.centroid_corner_distance(buildings)
        # buildings['centroid_corner_distance_mean'] = buildings_centroid_corner_distance['mean']
        # buildings['centroid_corner_distance_std'] = buildings_centroid_corner_distance['std']

    # Basic geometric properties
    if selected is None or selected['perimeter']:
        buildings['perimeter'] = buildings.geometry.length
    if selected is None or selected['shape_index']:
        buildings['shape_index'] = momepy.shape_index(buildings, momepy.longest_axis_length(buildings))
    if selected is None or selected['circular_compactness']:
        buildings['circular_compactness'] = momepy.circular_compactness(buildings)
    if selected is None or selected['square_compactness']:
        buildings['square_compactness'] = momepy.square_compactness(buildings)
    if selected is None or selected['weighted_axis_compactness']:
        buildings['weighted_axis_compactness'] = momepy.compactness_weighted_axis(buildings)
        # buildings['convexity'] = momepy.convexity(buildings)
    if selected is None or selected['courtyard_area']:
        buildings['courtyard_area'] = momepy.courtyard_area(buildings)
    if selected is None or selected['courtyard_index']:
        buildings['courtyard_index'] = momepy.courtyard_index(buildings)
    if selected is None or selected['fractal_dimension']:
        buildings['fractal_dimension'] = momepy.fractal_dimension(buildings)
    if selected is None or selected['facade_ratio']:
        buildings['facade_ratio'] = momepy.facade_ratio(buildings)

    # More complex morphological metrics
    if selected is None or selected['orientation']:
        buildings['orientation'] = momepy.orientation(buildings)
    if selected is None or selected['longest_axis_length']:
        buildings['longest_axis_length'] = momepy.longest_axis_length(buildings)
    if selected is None or selected['equivalent_rectangular_index']:
        buildings['equivalent_rectangular_index'] = momepy.equivalent_rectangular_index(buildings)
    if selected is None or selected['elongation']:
        buildings['elongation'] = momepy.elongation(buildings)
    if selected is None or selected['rectangularity']:
        buildings['rectangularity'] = momepy.rectangularity(buildings)
    if selected is None or selected['shared_walls']:
        buildings['shared_walls'] = momepy.shared_walls(buildings) / buildings['perimeter']
    if selected is None or selected['perimeter_wall']:
        buildings['perimeter_wall'] = momepy.perimeter_wall(buildings)  # TODO: check for possible parameters
        buildings['perimeter_wall'] = buildings['perimeter_wall'].fillna(0)
    # Metrics related to building adjacency and Graph

def generate_graph_building_metrics(buildings, streets, queen_1, selected=None):
    '''
    :param buildings:
    :param streets:
    :param tessellation:
    :param coplanar: switch to clique if there's overlapping polygons
    :return:
    '''
    queen_3 = queen_1.higher_order(k=3)
    build_q1 = graph.Graph.build_contiguity(buildings)
    
    if selected is None or selected['alignment']:
        buildings['alignment'] = momepy.alignment(buildings['orientation'], queen_1)
    if selected is None or selected['adjacency']:
        buildings['adjacency'] = momepy.building_adjacency(build_q1, queen_3) 
    if selected is None or selected['mean_interbuilding_distance']:
        buildings['mean_interbuilding_distance'] = momepy.mean_interbuilding_distance(buildings, queen_1, queen_3)
    if selected is None or selected['neighbour_distance']:
        buildings['neighbour_distance'] = momepy.neighbor_distance(buildings, queen_1)
    if selected is None or selected['courtyards_num']:
        buildings['courtyards_num'] = momepy.courtyards(buildings, build_q1)  # Calculate the number of courtyards within the joined structure

    if 'height' in buildings.keys():
        prof_metrics = ['width', 'openness', 'width_dev', 'height', 'height_dev', 'hw_ratio']
        heights = buildings['height']
    else:
        prof_metrics = ['width', 'openness', 'width_dev']
        heights = None
    streets[prof_metrics] = momepy.street_profile(streets, buildings, height=heights)
    if selected is not None:
        if not selected['width']:
            streets.drop(columns='width')
        if not selected['openness']:
            streets.drop(columns='openness')
        if not selected['width_dev']:
            streets.drop(columns='width_dev')

    if selected is None or selected['squareness']:
        buildings['street_alignment'] = momepy.street_alignment(buildings['orientation'], streets['str_orientation'], buildings["street_index"])
    return queen_3

def generate_tessellation_metrics(tessellations, buildings, selected=None):
    queen_1 = graph.Graph.build_contiguity(tessellations, rook=False)
    if selected is None or selected['tess_area']:
        tessellations['tess_area'] = tessellations.area
    if selected is None or selected['tess_circ_compactness']:
        tessellations['tess_circ_compactness'] = momepy.circular_compactness(tessellations)
    if selected is None or selected['tess_convexity']:
        tessellations['tess_convexity'] = momepy.convexity(tessellations)
    if selected is None or selected['tess_neighbors']:
        tessellations['tess_neighbors'] = momepy.neighbors(tessellations, queen_1, weighted=True)
    if selected is None or selected['tess_covered_area']:
        tessellations['tess_covered_area'] = queen_1.describe(tessellations.area)['sum']
    if selected is None or selected['tess_build_area_raio']:
        tessellations['tess_build_area_raio'] = buildings.area / tessellations.area
    if selected is None or selected['tess_orientation']:
        tessellations['tess_orientation'] = momepy.orientation(tessellations)
    if selected is None or selected['tess_building_orientation']:
        tessellations['tess_building_orientation'] = (tessellations['tess_orientation']-buildings['orientation']).abs()
    if selected is None or selected['tess_alignment']:
        tessellations['tess_alignment'] = momepy.alignment(tessellations['tess_orientation'], queen_1)
    if selected is None or selected['tess_equivalent_rect_index']:
        tessellations['tess_equivalent_rect_index'] = momepy.equivalent_rectangular_index(tessellations)
    return queen_1


def generate_streets_metrics(streets_gdf, selected=None):

    if selected is None or selected['str_orientation']:
        streets_gdf["str_orientation"] = momepy.orientation(streets_gdf)
    if selected is None or selected['str_longest_axis']:
        streets_gdf["str_longest_axis"] = momepy.longest_axis_length(streets_gdf)
        # streets_gdf["compactness_weighted_axis"] = momepy.compactness_weighted_axis(streets_gdf)
    if selected is None or selected['str_linearity']:
        streets_gdf["str_linearity"] = momepy.linearity(streets_gdf)
    if selected is None or selected['str_length']:
        streets_gdf["str_length"] = streets_gdf.length
    # Calculates natural continuity and hierarchy of street networks
    # coins = momepy.COINS(streets_gdf)
    # stroke_attr = coins.stroke_attribute()
    # streets_gdf["stroke_id"] = stroke_attr
    # # Group by stroke_id to calculate stroke-level continuity (total length of each stroke)
    # stroke_continuity = streets_gdf.groupby("stroke_id")["length"].sum().reset_index()
    # stroke_continuity.columns = ["stroke_id", "continuity"]
    # # Merge continuity back into the streets GeoDataFrame
    # streets_gdf = streets_gdf.merge(stroke_continuity, on="stroke_id", how="left")
    # # Rank strokes by length to create a hierarchy
    # stroke_continuity["hierarchy"] = stroke_continuity["continuity"].rank(
    #     ascending=False
    # )
    # # Merge hierarchy back into the streets GeoDataFrame
    # streets_gdf = streets_gdf.merge(
    #     stroke_continuity[["stroke_id", "hierarchy"]], on="stroke_id", how="left"
    # )

def generate_junctions_metrics(streets, verbose=False, selected=None):

    graph = momepy.gdf_to_nx(streets)

    graph = momepy.node_degree(graph)
    graph = momepy.closeness_centrality(graph, radius=5, distance="mm_len", verbose=verbose)
    # graph = momepy.meshedness(graph, radius=5, distance="mm_len")

    # Calculates clustering coefficient for each node (how connected are its neighbors)
    graph = momepy.clustering(graph)
    # Calculates cyclomatic complexity (number of loops) around each node
    graph = momepy.cyclomatic(graph, verbose=verbose)
    # Calculate the ratio of edges to nodes in a subgraph around each node
    graph = momepy.edge_node_ratio(graph, verbose=verbose)
    # Calculate the gamma index (connectivity) for each node
    graph = momepy.gamma(graph, verbose=verbose)
    # Calculate the mean node degree for a subgraph around each node
    graph = momepy.mean_node_degree(graph, verbose=verbose)
    # Calculate the mean distance to neighboring nodes for each node
    graph = momepy.mean_node_dist(graph, verbose=verbose)
    # Calculate the meshedness for subgraph around each node
    graph = momepy.meshedness(graph, verbose=verbose)
    # Calculate the density of nodes around each node
    graph = momepy.node_density(graph, radius=5, verbose=verbose)

    # Calculate the proportion of intersection types (special types of intersections)
    # junctions_metrics['proportion'] = momepy.proportion(street_network_graph)

    # Calculate straightness centrality for each node
    graph = momepy.straightness_centrality(graph, radius=5, verbose=verbose)

    nodes, edges = momepy.nx_to_gdf(graph)
    if selected is not None:
        if not selected['gamma']:
            nodes.drop(columns='gamma')
        if not selected['mean_nd']:
            nodes.drop(columns='mean_nd')
        if not selected['meanlen']:
            nodes.drop(columns='meanlen')
        if not selected['meshedness']:
            nodes.drop(columns='meshedness')
        if not selected['node_density']:
            nodes.drop(columns='node_density')
        if not selected['node_density_weighted']:
            nodes.drop(columns='node_density_weighted')
        if not selected['straightness']:
            nodes.drop(columns='straightness')
            
    ### TODO: drop unwanted metrics

    # Optional: Save the nodes with metrics to a GeoPackage file
    # nodes.to_file("nodes_with_metrics.gpkg", layer="nodes", driver="GPKG")
    return nodes, edges