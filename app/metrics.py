import geopandas as gpd
import momepy

from libpysal import graph

def generate_building_metrics(buildings: gpd.geodataframe, height_column_name=None):
    if height_column_name:
        buildings['Area'] = buildings.geometry.area
        buildings['floor_area'] = momepy.floor_area(buildings['Area'], buildings['building_height'])
        buildings['volume'] = momepy.volume(buildings['Area'], buildings['building_height'])
        buildings['form_factor'] = momepy.form_factor(buildings, buildings['building_height'])

    # Check if all geometries are either Polygon or MultiPolygon
    if (buildings.geometry.geom_type == 'Polygon').all():
        buildings['corners'] = momepy.corners(buildings)
        buildings['squareness'] = momepy.squareness(buildings)
        # buildings_centroid_corner_distance = momepy.centroid_corner_distance(buildings)
        # buildings['centroid_corner_distance_mean'] = buildings_centroid_corner_distance['mean']
        # buildings['centroid_corner_distance_std'] = buildings_centroid_corner_distance['std']

    # Basic geometric properties
    buildings['perimeter'] = buildings.geometry.length
    buildings['shape_index'] = momepy.shape_index(buildings, momepy.longest_axis_length(buildings))
    buildings['circular_compactness'] = momepy.circular_compactness(buildings)
    buildings['square_compactness'] = momepy.square_compactness(buildings)
    buildings['weighted_axis_compactness'] = momepy.compactness_weighted_axis(buildings)
    buildings['convexity'] = momepy.convexity(buildings)
    buildings['courtyard_area'] = momepy.courtyard_area(buildings)
    buildings['courtyard_index'] = momepy.courtyard_index(buildings)
    buildings['fractal_dimension'] = momepy.fractal_dimension(buildings)
    buildings['facade_ratio'] = momepy.facade_ratio(buildings)

    # More complex morphological metrics
    buildings['orientation'] = momepy.orientation(buildings)
    buildings['longest_axis_length'] = momepy.longest_axis_length(buildings)
    buildings['equivalent_rectangular_index'] = momepy.equivalent_rectangular_index(buildings)
    buildings['elongation'] = momepy.elongation(buildings)
    buildings['rectangularity'] = momepy.rectangularity(buildings)
    buildings['shared_walls_length'] = momepy.shared_walls(buildings) / buildings['perimeter']
    buildings['perimeter_wall'] = momepy.perimeter_wall(buildings)  # TODO: check for possible parameters
    buildings['perimeter_wall'] = buildings['perimeter_wall'].fillna(0)
    # Metrics related to building adjacency and Graph

    return buildings

def generate_graph_metrics(buildings, streets, tessellation, coplanar='raise',knnA=15, knnB=5):
    '''
    :param buildings:
    :param streets:
    :param tessellation:
    :param coplanar: switch to clique if there's overlapping polygons
    :return:
    '''
    delaunay = graph.Graph.build_triangulation(buildings.centroid, coplanar='clique').assign_self_weight()
    blg_orient = momepy.orientation(buildings)
    buildings['alignment'] = momepy.alignment(blg_orient, delaunay)

    knn_1 = graph.Graph.build_knn(buildings.centroid, k=knnA, coplanar='clique')  # adjust k if needed
    contiguity = graph.Graph.build_contiguity(buildings)
    buildings['adjacency'] = momepy.building_adjacency(contiguity, knn_1)  # TODO: check for queen1 and queen3
    buildings['mean_interbuilding_distance'] = momepy.mean_interbuilding_distance(buildings, delaunay, knn_1)  # TODO: check for queen1 and queen3
    buildings['neighbour_distance'] = momepy.neighbor_distance(buildings, delaunay)
    buildings['courtyards_num'] = momepy.courtyards(buildings,
                                                    contiguity)  # Calculate the number of courtyards within the joined structure

    if 'height' in buildings.keys():
        print('with height')
        prof_metrics = ['width', 'openness', 'width_dev', 'height', 'height_dev', 'hw_ratio']
        heights = buildings['height']
    else:
        print('no height')
        prof_metrics = ['width', 'openness', 'width_dev']
        heights = None
    streets[prof_metrics] = momepy.street_profile(streets, buildings, height=heights)

    blg_orient = momepy.orientation(buildings)
    str_orient = momepy.orientation(streets)
    buildings["street_index"] = momepy.get_nearest_street(buildings, streets)
    buildings['street_alignment'] = momepy.street_alignment(blg_orient, str_orient, buildings["street_index"])

    return buildings, streets


def generate_all_metrics(buildings, streets, tessellation, height_column_name=None):
    buildings = generate_building_metrics(buildings, height_column_name)
    # streets = generate_streets_metrics(streets)
    buildings, streets = generate_graph_metrics(buildings, streets, tessellation)
    return buildings, streets

def generate_streets_metrics(streets_gdf):
    streets_geometry = streets_gdf["geometry"]

    streets_gdf["orientation"] = momepy.orientation(streets_geometry)
    streets_gdf["longest_axis_length"] = momepy.longest_axis_length(streets_geometry)
    streets_gdf["compactness_weighted_axis"] = momepy.compactness_weighted_axis(
        streets_geometry
    )
    streets_gdf["linearity"] = momepy.linearity(streets_geometry)
    streets_gdf["length"] = streets_geometry.length
    # Calculates natural continuity and hierarchy of street networks
    coins = momepy.COINS(streets_gdf)
    stroke_attr = coins.stroke_attribute()
    # print(stroke_attr.head())
    streets_gdf["stroke_id"] = stroke_attr
    # Group by stroke_id to calculate stroke-level continuity (total length of each stroke)
    stroke_continuity = streets_gdf.groupby("stroke_id")["length"].sum().reset_index()
    stroke_continuity.columns = ["stroke_id", "continuity"]
    # Merge continuity back into the streets GeoDataFrame
    streets_gdf = streets_gdf.merge(stroke_continuity, on="stroke_id", how="left")
    # Rank strokes by length to create a hierarchy
    stroke_continuity["hierarchy"] = stroke_continuity["continuity"].rank(
        ascending=False
    )
    # Merge hierarchy back into the streets GeoDataFrame
    streets_gdf = streets_gdf.merge(
        stroke_continuity[["stroke_id", "hierarchy"]], on="stroke_id", how="left"
    )

def generate_junctions_metrics(streets, verbose=False):

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

    # Optional: Save the nodes with metrics to a GeoPackage file
    # nodes.to_file("nodes_with_metrics.gpkg", layer="nodes", driver="GPKG")
    return nodes, edges