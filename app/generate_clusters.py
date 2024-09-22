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

def plot_clusters_st(buildings):
    # Display message while plotting
    st.write("Plotting ...")

    # Create a figure and axis with a geographic projection
    fig, ax = plt.subplots(figsize=(12, 12))

    # Define colors for clusters
    colors = plt.get_cmap('tab20').colors

    # Plot buildings colored by cluster
    buildings.plot(column='cluster', cmap='Set1', legend=False, ax=ax, edgecolor='k')  # Add edgecolor for better visibility

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