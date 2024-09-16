import geopandas as gpd
import momepy
import pandas
import osmnx


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


if __name__ == "__main__":
    from data_input import load_roads_from_osm
    from preprocess import get_streets

    place = "Jerusalem, Israel"
    local_crs = 'EPSG:2039'
    network_type = 'drive'

    streets = load_roads_from_osm(place, network_type=network_type)
    streets, intersections = get_streets(streets=streets, local_crs=local_crs, get_nodes=True)

    # osm_graph = osmnx.graph_from_place(place, network_type="drive")
    # generate_junctions_metrics()
