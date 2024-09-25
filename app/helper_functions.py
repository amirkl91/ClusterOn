from rtree import index
import geopandas as gpd
from clustergram import Clustergram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from libpysal import graph
from packaging.version import Version
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from kneed import KneeLocator
from sklearn.cluster import KMeans



# Function to check which polygon in gdfB contains the polygon in gdfA
def _find_containing_polygon(polygon, B, spatial_index, column_to_add):
    # Get possible matches using bounding box intersection
    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = B.iloc[possible_matches_index]

    # Check for actual containment within those possible matches
    for idx, b_polygon in possible_matches.iterrows():
        if polygon.within(b_polygon.geometry):
            return idx if column_to_add == 'idx' else b_polygon[
                column_to_add]  # Return the index of the containing polygon in gdfB
    return None  # If no containing polygon is found


# Function to find the polygon in gdfB with the largest intersection area
def _find_largest_area_intersection(polygon, B, spatial_index, column_to_add):
    # Get possible matches using bounding box intersection
    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = B.iloc[possible_matches_index]

    max_area = 0
    best_match_idx = None

    # Check for actual intersection and find the largest area of intersection
    for idx, b_polygon in possible_matches.iterrows():
        intersection = polygon.intersection(b_polygon.geometry)
        if not intersection.is_empty:
            intersection_area = intersection.area
            if intersection_area > max_area:
                max_area = intersection_area
                best_match_idx = idx

    if column_to_add == 'idx' or best_match_idx is None:
        return best_match_idx
    return B.iloc[best_match_idx][
        column_to_add]  # Return the index of the polygon in gdfB with the largest intersection


def find_overlapping(gdfA, gdfB, column_to_add='idx', strictly_contained=False):
    """
    NOTE : make sure that gdfA and gdfB are in the same crs to avoid potential errors
    :param gdfA: the buildings dataFrame
    :param gdfB: the textures dataFrame
    :param column_to_add: the column from gdfB to add to the buildings dataFrame
    :param strictly_contained: if True, the function will check for strict containment, otherwise it will find the polygon in gdfB with the largest overlapping area
    :return: a new building dataFrame with the containing_polygon_index column
    """

    gdfA = gdfA.to_crs(gdfB.crs)

    # Create a spatial index for gdfB using R-tree
    spatial_index = index.Index()
    for idx, geometry in gdfB.iterrows():
        spatial_index.insert(idx, geometry.geometry.bounds)

    if strictly_contained:
        # Apply the function to find the containing polygon index for each polygon in gdfA
        gdfA['containing_polygon_' + column_to_add] = gdfA.geometry.apply(
            lambda x: _find_containing_polygon(x, gdfB, spatial_index, column_to_add))
    else:
        # Apply the function to find the index of the polygon in gdfB with the largest intersection
        gdfA[f'containing_polygon_' + column_to_add] = gdfA.geometry.apply(
            lambda x: _find_largest_area_intersection(x, gdfB, spatial_index, column_to_add))

    return gdfA


def find_group_leading_metrics(gdf, classification_column, n_components=2):
    group_leading_metrics = {}

    # Standardize the features
    scaler = StandardScaler()

    # Group by classification
    for group, group_data in gdf.groupby(classification_column):
        print(f"\nFinding leading metrics for group: {group}")

        # Drop the classification and geometry columns
        group_features = group_data.drop(columns=[classification_column, 'geometry'])

        # Standardize the group-specific data
        scaled_features = scaler.fit_transform(group_features)

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(scaled_features)

        # Get the principal components
        components = pca.components_

        # Create a DataFrame of components with feature names
        leading_metrics = pd.DataFrame(components, columns=group_features.columns)

        # Sort metrics by their absolute contribution to the principal components
        sorted_metrics = leading_metrics.apply(lambda x: x.abs().sort_values(ascending=False).index, axis=1)

        # Store the top metrics for this group
        group_leading_metrics[group] = sorted_metrics.head(2)  # Top 2 metrics per group

        # Print the top 2 metrics
        print(f"Top metrics for group {group}:")
        print(sorted_metrics.head(2))

    return group_leading_metrics



def find_overall_leading_metrics(gdf, classification_column):
    # Drop the classification and geometry columns for feature selection
    features = gdf.drop(columns=[classification_column, 'geometry'])
    target = gdf[classification_column]

    # Standardize the features (optional, but often helpful)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train a Random Forest model on the whole data
    model = RandomForestClassifier(random_state=42)
    model.fit(scaled_features, target)

    # Extract and sort feature importances
    feature_importances = pd.Series(model.feature_importances_, index=features.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Print the leading metrics
    print("Overall leading metrics for classification:")
    print(feature_importances.head(10))  # Top 10 most important metrics

    return feature_importances


# This is a modified version of the momepy.buffered_limit function that allows for adaptive buffer calculation on our data
def buffered_limit(gdf, buffer: float | str = 100, min_buffer: float = 0, max_buffer: float = 100, **kwargs, ):
    if buffer == "adaptive":

        gabriel = graph.Graph.build_triangulation(gdf.centroid, "gabriel", kernel="identity", coplanar='clique')
        max_dist = gabriel.aggregate("max")
        buffer = np.clip(max_dist / 2 + max_dist * 0.1, min_buffer, max_buffer).values

    elif not isinstance(buffer, int | float):
        raise ValueError("`buffer` must be either 'adaptive' or a number.")

    GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")
    return (
        gdf.buffer(buffer, **kwargs).union_all()
        if GPD_GE_10
        else gdf.buffer(buffer, **kwargs).unary_union
    )