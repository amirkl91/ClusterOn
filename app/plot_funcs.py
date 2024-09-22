import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_outliers(gdf, results, classification_column):
    """
    Function to plot the outliers from all clusters in different colors.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the base map (all buildings/streets)
    gdf.plot(ax=ax, color="lightgrey", label="All elements")

    # Collect all outliers from each cluster and plot them
    cmap = plt.get_cmap("tab10")  # Use a colormap for distinct colors
    cluster_colors = {}

    for i, cluster in enumerate(results.keys()):
        cluster_outliers = results[cluster]["outliers"]
        if cluster_outliers.empty:
            continue  # Skip clusters with no outliers
        cluster_outliers.plot(
            ax=ax, marker="o", color=cmap(i), label=f"Cluster {cluster} Outliers"
        )
        cluster_colors[cluster] = cmap(i)  # Store color for future use

    plt.title("Outliers for All Clusters")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_top_and_bottom_flexibility_scores(results):
    """
    Function to plot the top 5 and bottom 5 flexibility scores for each cluster.
    """
    for cluster, result in results.items():
        flexibility_score = result["flexibility_score"]

        # Get the top 5 and bottom 5 metrics by flexibility score
        top_5_metrics = flexibility_score.nlargest(5)
        bottom_5_metrics = flexibility_score.nsmallest(5)

        # Combine the top 5 and bottom 5 metrics for plotting
        selected_metrics = pd.concat([top_5_metrics, bottom_5_metrics])

        # Plot the selected metrics for this cluster
        plt.figure(figsize=(14, 8))
        bars = plt.bar(
            selected_metrics.index, selected_metrics, color="skyblue", width=0.8
        )

        # Highlight the top 5 flexible (green) and bottom 5 strict (red) metrics
        for idx in top_5_metrics.index:
            bars[selected_metrics.index.get_loc(idx)].set_color(
                "green"
            )  # Top 5 flexible
        for idx in bottom_5_metrics.index:
            bars[selected_metrics.index.get_loc(idx)].set_color(
                "red"
            )  # Bottom 5 strict

        # Rotate x-tick labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add title and labels
        plt.title(
            f"Top 5 and Bottom 5 Flexibility Scores for Cluster {cluster}", fontsize=16
        )
        plt.xlabel("Metrics")
        plt.ylabel("Flexibility Score (Avg of SD and Entropy)")

        # Show the plot
        plt.tight_layout()
        plt.show()


def plot_overall_leading_metrics(results):
    """
    Plot the overall leading metrics across all clusters based on flexibility score.
    """
    # Gather all flexibility scores
    combined_flexibility_scores = pd.Series(dtype=float)
    for cluster, result in results.items():
        combined_flexibility_scores = pd.concat(
            [combined_flexibility_scores, result["flexibility_score"]]
        )

    # Calculate overall flexibility score across all clusters
    overall_flexibility_score = combined_flexibility_scores.groupby(
        combined_flexibility_scores.index
    ).mean()

    # Get the top 5 and bottom 5 metrics across all clusters
    top_5_overall = overall_flexibility_score.nlargest(5)
    bottom_5_overall = overall_flexibility_score.nsmallest(5)

    # Combine the top 5 and bottom 5 metrics for plotting
    overall_selected_metrics = pd.concat([top_5_overall, bottom_5_overall])

    # Plot the overall selected metrics
    plt.figure(figsize=(14, 8))
    bars = plt.bar(
        overall_selected_metrics.index,
        overall_selected_metrics,
        color="skyblue",
        width=0.8,
    )

    # Highlight the top 5 flexible (green) and bottom 5 strict (red) metrics
    for idx in top_5_overall.index:
        bars[overall_selected_metrics.index.get_loc(idx)].set_color(
            "green"
        )  # Top 5 flexible
    for idx in bottom_5_overall.index:
        bars[overall_selected_metrics.index.get_loc(idx)].set_color(
            "red"
        )  # Bottom 5 strict

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add title and labels
    plt.title("Overall Top 5 and Bottom 5 Flexibility Scores", fontsize=16)
    plt.xlabel("Metrics")
    plt.ylabel("Flexibility Score (Avg of SD and Entropy)")

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_overall_vif(results):
    """
    Function to plot the overall VIF (Variance Inflation Factor) across all clusters.
    """
    # Gather all VIF data from each cluster
    vif_df = pd.DataFrame()

    for cluster, result in results.items():
        vif_cluster = result["vif"]
        vif_df = pd.concat([vif_df, vif_cluster.set_index("feature")], axis=1)

    # Calculate the mean VIF for each feature across all clusters
    vif_mean = vif_df.mean(axis=1)

    # Plot the overall VIF
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(vif_mean.index, vif_mean, color="purple")
    ax.set_title("Overall Variance Inflation Factor (VIF) Across All Clusters")
    ax.set_xticklabels(vif_mean.index, rotation=45, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Mean VIF Across Clusters")
    plt.tight_layout()
    plt.show()


def plot_overall_metrics_influence(results):
    """
    Function to plot the overall metrics that influenced outliers the most across all clusters.
    """
    # Gather metrics influence data from each cluster
    metrics_influence_df = pd.DataFrame()

    for cluster, result in results.items():
        metrics_influence = result["metrics_influence"]
        metrics_influence_df = pd.concat(
            [metrics_influence_df, metrics_influence], axis=1
        )

    # Sum or average the influence scores for each metric across all clusters
    # You can either sum or average, depending on what makes sense for your analysis
    overall_influence = metrics_influence_df.sum(axis=1)  # Summing across clusters

    # Plot the overall metrics that influenced outliers the most
    fig, ax = plt.subplots(figsize=(10, 6))
    overall_influence.plot(kind="bar", ax=ax, color="green")
    ax.set_title(
        "Overall Metrics That Influenced Outliers the Most Across All Clusters"
    )
    plt.xlabel("Metrics")
    plt.ylabel("Total Influence Across Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_cluster_similarity(silhouette_values, target, silhouette_avg):
    """
    Function to plot the silhouette values for clusters based on precomputed metrics.
    Args:
        silhouette_values: Array-like, silhouette values for each data point.
        target: Array-like, cluster labels.
        silhouette_avg: The average silhouette score for the entire dataset.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    y_lower = 10

    # Loop over each cluster to plot silhouette values
    for i in np.unique(target):
        ith_cluster_silhouette_values = silhouette_values[target == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # Add space between clusters

    # Plot the average silhouette score as a vertical line
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_title("Silhouette plot for each cluster")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    plt.show()
