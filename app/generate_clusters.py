import warnings
import geopandas
import libpysal
import momepy
import osmnx
import pandas as pd
import configparser
import os
from clustergram import Clustergram
import matplotlib.pyplot as plt
from bokeh.plotting import show
import matplotlib.patches as mpatches
import streamlit as st
from kneed import KneeLocator
from sklearn.cluster import KMeans
import geopandas as gpd
from collections import Counter


def _elbow(gdf, K: range):
    """
    :param gdf: the dataFrame to cluster
    :param K: iterable of number of clusters to try i.g range(1, 10)
    :param plot: whether to plot the distortion and inertia or not
    :return:
    """
    distortions = []
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(gdf.fillna(0))
        inertia = kmeanModel.inertia_
        distortions.append(inertia / len(gdf))

    return distortions


def _clusters_scores(gdf: gpd.GeoDataFrame, model='kmeans', standardize=True, min_clusters=1,
                     max_clusters=15,
                     n_init=13, random_state=42, plot=False) -> pd.DataFrame:
    """
    :param gdf: geoDataFrame that contains the data
    :param model: model to use for clustering ['kmeans', 'gmm', 'minibatchkmeans', 'hierarchical']
    :param max_clusters: maximum number of clusters to consider
    :param min_clusters: minimum number of clusters to consider
    :param standardize: whether to standardize the data or not
    :return: most suitable number of clusters
    """
    scores = {'K': [i for i in range(min_clusters, max_clusters + 1) if i > 1]}
    if standardize:
        gdf = (gdf - gdf.mean()) / gdf.std()
    K = range(min_clusters, max_clusters + 1)

    cgram = Clustergram(K, method=model, n_init=n_init, random_state=random_state)
    cgram.fit(gdf.fillna(0))

    scores['silhouette'] = cgram.silhouette_score()
    scores['davies_bouldin'] = cgram.davies_bouldin_score()
    scores['calinski_harabasz'] = cgram.calinski_harabasz_score()
    return pd.DataFrame(scores)


def select_best_num_of_clusters(gdf: gpd.GeoDataFrame, model='kmeans', standardize=True, min_clusters=1,
                                max_clusters=15,
                                n_init=13, random_state=42, plot=False) -> int:
    """
    :param gdf: geoDataFrame that contains the data
    :param model: model to use for clustering ['kmeans', 'gmm', 'minibatchkmeans', 'hierarchical']
    :param max_clusters: maximum number of clusters to consider
    :param min_clusters: minimum number of clusters to consider
    :param standardize: whether to standardize the data or not
    :return: most suitable number of clusters
    """
    best_scores = {}
    scores = _clusters_scores(gdf, model, standardize, min_clusters, max_clusters, n_init, random_state)
    K = range(min_clusters, max_clusters + 1)
    distortions = _elbow(gdf, K)

    best_scores['distortion'] = KneeLocator(K, distortions, curve='convex', direction='decreasing').elbow

    if plot:
        plt.plot(K, distortions, 'bx-')
        plt.vlines(best_scores['distortion'], plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        plt.xlabel('Number of clusters')
        plt.ylabel('distortion')
        plt.title(f'Elbow at k = {best_scores['distortion']}')
        plt.show()

    best_scores['silhouette'] = scores.loc[scores['K'] == scores['silhouette'].idxmax()]['K'].values[0]
    best_scores['davies_bouldin'] = scores.loc[scores['K'] == scores['davies_bouldin'].idxmin()]['K'].values[0]
    best_scores['calinski_harabasz'] = scores.loc[scores['K'] == scores['calinski_harabasz'].idxmax()]['K'].values[0]
    print(scores)
    print(best_scores)
    return Counter(list(best_scores.values())).most_common(1)[0][0]


def get_cgram(standardized, max_range):
    cgram = Clustergram(range(1, max_range), n_init=10, random_state=42)
    cgram.fit(standardized.fillna(0))
    show(cgram.bokeh())
    cgram.labels.head()
    return cgram

def add_cluster_col(merged, buildings, cgram, clusters_num):
    merged["cluster"] = cgram.labels[clusters_num].values
    buildings["cluster"] = merged["cluster"]
    return buildings

def plot_clusters(buildings):
    # Define the colors for each cluster
    colors = plt.get_cmap('tab20').colors
    categories = buildings['cluster'].unique()

    # Create the legend handles
    legend_handles = [mpatches.Patch(color=colors[i], label=f'Cluster {category}') 
                    for i, category in enumerate(categories)]

    # Create separate plots for each attribute
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot for 'cluster' attribute
    buildings.plot(column='cluster', cmap='Set1', legend=False, ax=ax)

    # Add title and customize plot
    ax.set_title('Urban Types by Cluster', fontsize=16)
    ax.set_axis_off()  # Optionally remove axis lines for a cleaner look

    # Add the custom legend
    ax.legend(handles=legend_handles, title='Cluster', bbox_to_anchor=(1, 1), loc='upper left')

    # Display the plot
    plt.show()

def plot_clusters_st(buildings):
    # Display message while plotting
    st.write("Plotting ...")

    # Create a figure and axis with a geographic projection
    fig, ax = plt.subplots(figsize=(12, 12))

    # Define colors for clusters
    colors = plt.get_cmap('tab20').colors

    # Plot buildings colored by cluster
    buildings.plot(column='cluster', cmap='Set1', legend=False, ax=ax)  # Add edgecolor for better visibility

    # Add title and customize plot
    ax.set_title('Urban Types by Cluster', fontsize=16)
    ax.set_axis_off()  # Optionally remove axis lines for a cleaner look

    # Create legend handles
    categories = buildings['cluster'].unique()
    legend_handles = [mpatches.Patch(color=colors[i % len(colors)], label=f'Cluster {category}') 
                      for i, category in enumerate(categories)]

    # Add the custom legend
    ax.legend(handles=legend_handles, title='Cluster', bbox_to_anchor=(1, 1), loc='upper left')

    # Show plot in Streamlit
    st.pyplot(fig)