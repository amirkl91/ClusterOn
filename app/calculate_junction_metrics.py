import geopandas as gpd
import momepy
import pandas
import osmnx


def generate_junctions_metrics(osmnx_graph, local_crs=None):
    streets = osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(osmnx_graph),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    ).reset_index(drop=True)
    streets = momepy.remove_false_nodes(streets)
    streets = streets[["geometry"]]
    # profile = momepy.street_profile(streets, buildings)
    # streets[profile.columns] = profile
    graph = momepy.gdf_to_nx(streets)
    graph = momepy.node_degree(graph)
    graph = momepy.closeness_centrality(graph, radius=5, distance="mm_len")
    # graph = momepy.meshedness(graph, radius=5, distance="mm_len")

    # Calculates clustering coefficient for each node (how connected are its neighbors)
    graph = momepy.clustering(graph)

    # Calculates cyclomatic complexity (number of loops) around each node
    graph = momepy.cyclomatic(graph)

    # Calculate the ratio of edges to nodes in a subgraph around each node
    graph = momepy.edge_node_ratio(graph)

    # Calculate the gamma index (connectivity) for each node
    graph = momepy.gamma(graph)

    # Calculate the mean node degree for a subgraph around each node
    graph = momepy.mean_node_degree(graph)

    # Calculate the mean distance to neighboring nodes for each node
    graph = momepy.mean_node_dist(graph)

    # Calculate the meshedness for subgraph around each node
    graph = momepy.meshedness(graph)

    # Calculate the degree of each node
    graph = momepy.node_degree(graph)

    # Calculate the density of nodes around each node
    graph = momepy.node_density(graph, radius=5)

    # Calculate the proportion of intersection types (special types of intersections)
    # junctions_metrics['proportion'] = momepy.proportion(street_network_graph)

    # Calculate straightness centrality for each node
    graph = momepy.straightness_centrality(graph, radius=5)

    nodes, edges = momepy.nx_to_gdf(graph)
    print(nodes.head())  # Check the node metrics
    print(edges.head())  # Check the edges

    # Optional: Save the nodes with metrics to a GeoPackage file
    # nodes.to_file("nodes_with_metrics.gpkg", layer="nodes", driver="GPKG")
    return nodes


if __name__ == "__main__":
    place = "Jerusalem, Israel"

    osm_graph = osmnx.graph_from_place(place, network_type="drive")
    generate_junctions_metrics()
