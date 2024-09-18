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

## config part

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
