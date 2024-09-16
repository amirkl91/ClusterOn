import geopandas as gpd
import momepy

from libpysal import graph
from libpysal import graph


def building_metrics(buildings: gpd.geodataframe, height_column_name=None):
    buildings["area"] = buildings.area
    if height_column_name is not None:
        buildings["floor_area"] = momepy.floor_area(
            buildings["area"], buildings[height_column_name]
        )
        buildings["volume"] = momepy.volume(
            buildings["area"], buildings[height_column_name]
        )
        buildings["form_factor"] = momepy.form_factor(
            buildings, buildings[height_column_name]
        )

    # Basic geometric properties
    buildings["perimeter"] = buildings.geometry.length
    buildings["shape_index"] = momepy.shape_index(
        buildings, momepy.longest_axis_length(buildings)
    )
    buildings["circular_compactness"] = momepy.circular_compactness(buildings)
    buildings["square_compactness"] = momepy.square_compactness(buildings)
    buildings["weighted_axis_compactness"] = momepy.compactness_weighted_axis(buildings)
    buildings["convexity"] = momepy.convexity(buildings)
    buildings["courtyard_area"] = momepy.courtyard_area(buildings)
    buildings["courtyard_index"] = momepy.courtyard_index(buildings)
    buildings["corners"] = momepy.corners(
        buildings, include_interiors=True
    )  # include_interiors=False works only for polygons not multipolygons
    buildings["fractal_dimension"] = momepy.fractal_dimension(buildings)
    buildings["facade_ratio"] = momepy.facade_ratio(buildings)
    buildings[["centr_corn_dist_mean", "centr_corn_dist_std"]] = (
        momepy.centroid_corner_distance(buildings)
    )

    # More complex morphological metrics
    buildings["orientation"] = momepy.orientation(buildings)
    buildings["longest_axis_length"] = momepy.longest_axis_length(buildings)
    buildings["equivalent_rectangular_index"] = momepy.equivalent_rectangular_index(
        buildings
    )
    buildings["elongation"] = momepy.elongation(buildings)
    buildings["rectangularity"] = momepy.rectangularity(buildings)
    buildings["squareness"] = momepy.squareness(
        buildings, include_interiors=True
    )  # without interiors works only for polygons not multipolygons
    buildings["shared_walls_length"] = (
        momepy.shared_walls(buildings) / buildings["perimeter"]
    )
    buildings["perimeter_wall"] = momepy.perimeter_wall(
        buildings
    )  # TODO: check for possible parameters
    buildings["perimeter_wall"] = buildings["perimeter_wall"].fillna(0)


def building_graph_metrics(buildings):
    # Metrics related to building adjacency and Graph

    delaunay = graph.Graph.build_triangulation(
        buildings.centroid, coplanar="clique"
    ).assign_self_weight()
    orientation = momepy.orientation(buildings)
    buildings["alignment"] = momepy.alignment(orientation, delaunay)

    knn15 = graph.Graph.build_knn(
        buildings.centroid, k=15, coplanar="clique"
    )  # adjust k if needed
    contiguity = graph.Graph.build_contiguity(buildings)
    buildings["adjacency"] = momepy.building_adjacency(
        contiguity, knn15
    )  # TODO: check for queen1 and queen3
    buildings["mean_interbuilding_distance"] = momepy.mean_interbuilding_distance(
        buildings, delaunay, knn15
    )  # TODO: check for queen1 and queen3
    buildings["neighbour_distance"] = momepy.neighbor_distance(buildings, delaunay)
    buildings["courtyards_num"] = momepy.courtyards(
        buildings, contiguity
    )  # Calculate the number of courtyards within the joined structure
    return contiguity


def building_tess_metrics(buildings, tessellations, contiguity):
    ### Has wierd errors. Dissregard for now
    # metrics related to tessellation
    tess_orient = momepy.orientation(tessellations)
    buildings["cell_orientation"] = momepy.cell_alignment(
        buildings["orientation"], tess_orient
    )
    buildings["num_of_neighbours"] = momepy.neighbors(
        tessellations, contiguity, weighted=True
    )

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

def generate_junctions_metrics(streets, local_crs=None, verbose=False):

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