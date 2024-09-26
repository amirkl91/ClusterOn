import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import streamlit as st


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


def plot_flexibility_scores_table(results):
    """
    Function to plot a table showing the top 5 and bottom 5 flexibility scores for each cluster,
    with separate columns for the metric names and values, and a title above the table.
    """
    # Prepare data for the table
    table_data = []
    # Iterate over each cluster's results
    for cluster, result in results.items():
        flexibility_score = result["flexibility_score"]

        # Get the top 5 and bottom 5 metrics by flexibility score
        top_5_metrics = flexibility_score.nlargest(5)
        bottom_5_metrics = flexibility_score.nsmallest(5)

        # Add top 5 and bottom 5 metrics as separate columns for metric names and values
        top_5_metrics_names = "\n".join(top_5_metrics.index)
        top_5_metrics_values = "\n".join(
            [f"{score:.2f}" for score in top_5_metrics.values]
        )
        bottom_5_metrics_names = "\n".join(bottom_5_metrics.index)
        bottom_5_metrics_values = "\n".join(
            [f"{score:.2f}" for score in bottom_5_metrics.values]
        )

        # Add the data to the table
        table_data.append(
            [
                f"Cluster {cluster}",
                top_5_metrics_names,
                top_5_metrics_values,
                bottom_5_metrics_names,
                bottom_5_metrics_values,
            ]
        )

    # Create a table figure with a larger height and width to fit the content
    fig, ax = plt.subplots(
        figsize=(13, len(results) * 1.5)
    )  # Increase figure size for better visibility
    ax.axis("tight")
    ax.axis("off")

    # Define table headers and content
    headers = [
        "Cluster",
        "Top 5 Flexible Metrics",
        "Top 5 Flexibility Scores",
        "Bottom 5 Strict Metrics",
        "Bottom 5 Strict Scores",
    ]
    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )

    # Adjust the font size and layout for better visibility
    table.auto_set_font_size(True)

    # Adjust column width to reduce the size of the 'Cluster' column
    table.auto_set_column_width([0, 1, 2, 3, 4])
    table.column_widths = [
        0.07,
        0.3,
        0.1,
        0.3,
        0.1,
    ]  # Adjust column widths manually if needed

    # Set header row colors
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:  # Header row
            cell.set_fontsize(12)
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("darkblue")
            cell.set_edgecolor("white")
        elif row % 2 == 0:  # Alternate row colors for visibility
            cell.set_facecolor(mcolors.CSS4_COLORS["lightgrey"])

    # Set font sizes for rows and adjust height for better readability of multi-line content
    for row in table._cells:
        table._cells[row].set_height(0.13)  # Adjust this value if necessary

    # Adjust the layout to ensure the table fills the figure and moves the title higher
    plt.subplots_adjust(top=0.95)  # Move title higher

    # Add title separately above the table
    plt.suptitle(
        "Top 5 and Bottom 5 Flexibility Scores for Each Cluster", fontsize=18, y=0.95
    )

    plt.tight_layout()
    plt.show()


def save_flexibility_scores_to_csv(results, output_filename="flexibility_scores.csv"):
    """
    Function to save the top 5 and bottom 5 flexibility scores for each cluster into a CSV file,
    sorted by flexibility score in decreasing order.
    Args:
        results: Dictionary containing the flexibility scores for each cluster.
        output_filename: Name of the CSV file to save the results.
    """
    # Prepare a list to store flexibility scores
    data_to_save = []

    # Iterate over each cluster's results
    for cluster, result in results.items():
        flexibility_score = result["flexibility_score"]

        # Sort flexibility scores in decreasing order
        sorted_scores = flexibility_score.sort_values(ascending=False)

        # Add each score and corresponding metric to the list
        for metric, score in sorted_scores.items():
            data_to_save.append([cluster, metric, score])

    # Convert the list to a DataFrame
    flexibility_df = pd.DataFrame(
        data_to_save, columns=["Cluster", "Metric", "Flexibility Score"]
    )

    # Save the DataFrame to a CSV file
    flexibility_df.to_csv(output_filename, index=False)


def plot_top_important_metrics(feature_importances):
    plt.figure(figsize=(10, 6))
    feature_importances.head(10).plot(
        kind="bar", color="skyblue"
    )  # Plot the top 10 features
    plt.title("Top 10 metrics that influence the classification the most")
    plt.ylabel("Feature Importance")
    plt.xlabel("Metrics")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_overall_flexibility(results):
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
    ax.set_title("Overall (VIF) - how a metric is correlated to all others")
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


#########################################streamlit#############################


def streamlit_plot_top_important_metrics(feature_importances):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the top 10 features
    feature_importances.head(10).plot(kind="bar", color="skyblue", ax=ax)

    # Set the title and labels
    ax.set_title("Top 10 metrics that influence the classification the most")
    ax.set_ylabel("Feature Importance")
    ax.set_xlabel("Metrics")

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout for a clean look
    plt.tight_layout()

    # Use Streamlit to display the plot
    st.pyplot(fig)


def streamlit_plot_flexibility_scores_table(results):
    """
    Function to display a table showing the top 5 and bottom 5 flexibility scores for each cluster
    in Streamlit.
    """
    # Prepare data for the table
    table_data = []

    # Iterate over each cluster's results
    for cluster, result in results.items():
        flexibility_score = result["flexibility_score"]

        # Get the top 5 and bottom 5 metrics by flexibility score
        top_5_metrics = flexibility_score.nlargest(5)
        bottom_5_metrics = flexibility_score.nsmallest(5)

        # Add top 5 and bottom 5 metrics as separate columns for metric names and values
        top_5_metrics_names = ", ".join(top_5_metrics.index)
        top_5_metrics_values = ", ".join(
            [f"{score:.2f}" for score in top_5_metrics.values]
        )
        bottom_5_metrics_names = ", ".join(bottom_5_metrics.index)
        bottom_5_metrics_values = ", ".join(
            [f"{score:.2f}" for score in bottom_5_metrics.values]
        )

        # Add the data to the table
        table_data.append(
            {
                "Cluster": f"Cluster {cluster}",
                "Top 5 Flexible Metrics": top_5_metrics_names,
                "Top 5 Flexibility Scores": top_5_metrics_values,
                "Bottom 5 Strict Metrics": bottom_5_metrics_names,
                "Bottom 5 Strict Scores": bottom_5_metrics_values,
            }
        )

    # Convert the table data to a DataFrame
    df = pd.DataFrame(table_data)

    # Display the table in Streamlit
    st.write("### Top 5 and Bottom 5 Flexibility Scores for Each Cluster")
    st.dataframe(df)


def streamlit_plot_outliers(gdf, results, classification_column):
    """
    Function to plot the outliers from all clusters in different colors, adapted for Streamlit.
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

    # Use Streamlit to display the plot
    st.pyplot(fig)


def streamlit_plot_overall_vif(results):
    """
    Function to plot the overall VIF (Variance Inflation Factor) across all clusters, adapted for Streamlit.
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
    ax.set_title("Overall (VIF) - how a metric is correlated to all others")
    ax.set_xticklabels(vif_mean.index, rotation=45, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Mean VIF Across Clusters")
    plt.tight_layout()

    # Use Streamlit to display the plot
    st.pyplot(fig)
