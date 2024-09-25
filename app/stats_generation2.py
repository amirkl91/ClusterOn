import geopandas as gpd
import matplotlib.pyplot as plt

# from matplotlib.lines import Line2D
# from shapely.geometry import Point
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score

# from esda.moran import Moran
# import libpysal
# from esda.getisord import G
# from sklearn.cluster import KMeans
import geopandas as gpd

# from scipy.stats import skew, kurtosis, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from sklearn.manifold import TSNE
# import umap.umap_ as umap
# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# from scipy.stats import entropy
# import shap
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score

# import os
import plot_funcs as pf

local_crs = "EPSG:2039"

"""
for each classification:
- basic stats: mean, variance, min, max, 
- outliers by and then use their geometry to plot
- leading metrics by PCA
- correlation between metrics in the classification
- Find LocalOutlierFactor

for global analysis:
- corr matrix, corralation between metrics.  
- how well seperated are the clusters
- find global ouliers and find the clusters that contain most of the ouliers
- find local outliers, using LocalOutlierFactor
"""


def analyze_gdf(gdf, classification_column, csv_folder_path):
    cluster_results = {}
    # Loop over each cluster in the classification column
    for cluster in gdf[classification_column].unique():
        cluster_rows = gdf[gdf[classification_column] == cluster]
        cluster_rows = cluster_rows.drop(columns=[classification_column])
        cluster_results[cluster] = analyze_cluster(cluster_rows)
        # df = output_cluster_stats(cluster, cluster_results)
        # df.to_csv(f"{csv_folder_path}/cluster{cluster}_analysis.csv", index=True)
    # Perform global analysis
    global_summary = perform_global_analysis(
        gdf, classification_column, cluster_results
    )
    # Plot relevant information
    plot_all_cluster_results(
        gdf, cluster_results, classification_column, global_summary
    )
    # Add another layers to the gdf
    gdf = classify_outliers(gdf, cluster_results)
    outlier_counts = gdf.groupby(classification_column)["outlier_flag"].sum()
    print(outlier_counts)
    return gdf, cluster_results


def perform_global_analysis(gdf, classification_column, cluster_results):
    """
    Perform global analysis like SVD, leading metrics, and similarity analysis.
    """
    numeric_columns = gdf.select_dtypes(include=[float, int]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gdf[numeric_columns])

    # Supervised analysis to find the leading metrics for classification
    supervised_importances = supervised_leading_metrics(gdf, classification_column)

    # Cluster similarity analysis
    similarity_results = cluster_similarity_analysis(gdf, classification_column)
    # Find clusters with highest and lowest outlier ratios
    highest_outlier_cluster = max(
        cluster_results, key=lambda x: cluster_results[x]["outlier_ratio"]
    )
    lowest_outlier_cluster = min(
        cluster_results, key=lambda x: cluster_results[x]["outlier_ratio"]
    )
    print(f"Silhouette and Davies-Bouldin scores: {similarity_results}")
    print(
        f"Cluster with highest outlier ratio: {highest_outlier_cluster}, "
        f"Ratio: {cluster_results[highest_outlier_cluster]['outlier_ratio']:.2f}"
    )
    print(
        f"Cluster with lowest outlier ratio: {lowest_outlier_cluster}, "
        f"Ratio: {cluster_results[lowest_outlier_cluster]['outlier_ratio']:.2f}"
    )

    # Return global summary
    global_summary = {
        "supervised_importances": supervised_importances,
        "similarity_results": similarity_results,
        "highest_outlier_cluster": highest_outlier_cluster,
        "lowest_outlier_cluster": lowest_outlier_cluster,
    }

    return global_summary


def plot_all_cluster_results(
    gdf, cluster_results, classification_column, global_summary
):
    """
    Function to plot all relevant information about clusters: outliers, flexibility, VIF, and metrics influence.
    """
    # Plot outliers for each cluster
    pf.plot_outliers(gdf, cluster_results, classification_column)
    # Plot top 5 and bottom 5 flexibility scores for each cluster
    pf.plot_flexibility_scores_table(cluster_results)

    # # Plot overall VIF for all clusters
    # pf.plot_overall_vif(cluster_results)

    # Plot overall metrics that influenced outliers the most
    pf.plot_overall_metrics_influence(cluster_results)

    pf.save_flexibility_scores_to_csv(cluster_results)
    # Plot cluster similarities
    # silhouette_avg, silhouette_values, target = global_summary["similarity_results"]
    # pf.plot_cluster_similarity(silhouette_values, target, silhouette_avg)


def analyze_cluster(gdf):
    result = {}
    numeric_columns = gdf.select_dtypes(
        include=[float, int]
    ).columns  # Only numeric columns

    # 1. Basic statistics: mean, variance, min, max
    stats = gdf[numeric_columns].describe().T
    stats["variance"] = gdf[numeric_columns].var()  # Variance
    result["basic_stats"] = stats

    # normalize
    numeric_data = gdf.select_dtypes(include=[float, int])
    geometry_data = gdf["geometry"]  # Keep geometry column
    # Standardize numeric columns
    scaler = StandardScaler()
    numeric_data = pd.DataFrame(
        scaler.fit_transform(numeric_data), columns=numeric_data.columns
    )
    # Reattach the geometry column to the standardized numeric data
    gdf = gpd.GeoDataFrame(numeric_data, geometry=geometry_data)

    # 2. Outlier detection using the IQR method (outliers are 1.5 * IQR beyond Q1 and Q3)
    Q1 = gdf[numeric_columns].quantile(0.25)
    Q3 = gdf[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (
        (gdf[numeric_columns] < (Q1 - 1.5 * IQR))
        | (gdf[numeric_columns] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    result["outliers"] = gdf[outliers]  # Store rows that are outliers

    # 2a. Calculate the outlier ratio (number of outliers / total points)
    outlier_ratio = outliers.sum() / len(gdf)
    result["outlier_ratio"] = outlier_ratio  # Store the outlier ratio

    # 3. Identify which metrics contributed most to the outliers
    metrics_influence = (gdf[numeric_columns] > (Q3 + 1.5 * IQR)) | (
        gdf[numeric_columns] < (Q1 - 1.5 * IQR)
    )
    result["metrics_influence"] = metrics_influence.sum().sort_values(ascending=False)

    # 5. Correlation analysis between metrics
    correlation_matrix = gdf[numeric_columns].corr()
    result["correlation_matrix"] = correlation_matrix

    # 6. Local Outlier Factor (LOF)

    lof = LocalOutlierFactor(n_neighbors=5)
    lof_scores = lof.fit_predict(gdf[numeric_columns])
    lof_results = pd.DataFrame(
        {
            "lof_score": lof_scores,
            "is_outlier": lof_scores == -1,  # Flag outliers (-1 indicates an outlier)
        }
    )
    lof_outliers = gdf[lof_results["is_outlier"] == True]  # Local outliers
    result["lof_outliers"] = lof_outliers

    # 7. Variance Inflation Factor (VIF) - scale the data first
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gdf[numeric_columns])

    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_columns
    vif_data["VIF"] = [
        variance_inflation_factor(scaled_data, i) for i in range(scaled_data.shape[1])
    ]
    result["vif"] = vif_data

    # 8. Calculate Flexibility Score
    stds = stats["std"]
    sorted_stds = stds.sort_values()
    entropy_weights = entropy_weighting(gdf)
    sorted_entropy = entropy_weights.sort_values()

    normalized_stds = (sorted_stds - sorted_stds.min()) / (
        sorted_stds.max() - sorted_stds.min()
    )
    normalized_entropy = (sorted_entropy - sorted_entropy.min()) / (
        sorted_entropy.max() - sorted_entropy.min()
    )

    flexibility_score = (normalized_stds + normalized_entropy) / 2
    result["flexibility_score"] = flexibility_score

    return result


def plot_cluster_analysis(gdf, results, cluster_number):
    # Get the numeric columns again
    numeric_columns = gdf.select_dtypes(include=[float, int]).columns

    # 1. Plot basic statistics (mean, variance) in a separate figure
    fig, ax = plt.subplots(figsize=(10, 6))
    stats = results["basic_stats"]
    ax.bar(stats.index, stats["mean"], yerr=stats["variance"], capsize=5, color="blue")
    ax.set_title(f"Mean and Variance of Metrics for Cluster {cluster_number}")
    ax.set_xticklabels(stats.index, rotation=45, ha="right")
    plt.show()

    # 2. Plot correlation matrix in a separate figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(results["correlation_matrix"], annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(f"Correlation Matrix of Metrics for Cluster {cluster_number}")
    plt.show()

    # 3. Plot the outliers (IQR) and LOF outliers in the same figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot IQR outliers
    base = gdf.plot(ax=axes[0], color="lightgrey")
    results["outliers"].plot(ax=base, color="red", marker="o", label="Outliers")
    axes[0].set_title(f"Outliers Detected by IQR for Cluster {cluster_number}")

    # Plot LOF outliers
    base = gdf.plot(ax=axes[1], color="lightgrey")
    results["lof_outliers"].plot(
        ax=base, color="orange", marker="x", label="LOF Outliers"
    )
    axes[1].set_title(
        f"Local Outlier Factor (LOF) Outliers for Cluster {cluster_number}"
    )

    plt.tight_layout()
    plt.show()

    # 4. Plot VIF (Variance Inflation Factor) in a separate figure with explanation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results["vif"]["feature"], results["vif"]["VIF"], color="purple")
    ax.set_title(f"Variance Inflation Factor (VIF) - Cluster {cluster_number}")
    ax.set_xticklabels(results["vif"]["feature"], rotation=45, ha="right")
    plt.show()

    # 5. Plot Maximum and Minimum values as text only with adjusted spacing
    fig, ax = plt.subplots(figsize=(10, 6))
    max_values = gdf[numeric_columns].max()
    min_values = gdf[numeric_columns].min()
    ind = np.arange(len(numeric_columns))

    # Adjust vertical spacing with slightly more space to avoid overlap
    y_positions = np.linspace(
        0.9, 0.1, len(numeric_columns)
    )  # Adjust range to distribute better

    # Loop through the metrics and display the values as text
    for i, metric in enumerate(numeric_columns):
        # Create a formatted string with the metric, max, and min values
        text = (
            f"{metric}:\nMax: {round(max_values[i], 2)}, Min: {round(min_values[i], 2)}"
        )

        # Display the text on the plot, adjust the y_positions for spacing
        ax.text(0.5, y_positions[i], text, fontsize=12, ha="center", va="center")

    # Hide the axis since it's just text
    ax.set_axis_off()

    # Set title and display
    plt.title(f"Maximum and Minimum Values for Cluster {cluster_number}")
    plt.tight_layout()
    plt.show()


def entropy_weighting(cluster_gdf):
    numeric_columns = cluster_gdf.select_dtypes(include=[float, int]).columns
    n = len(cluster_gdf)
    entropy = {}
    for col in numeric_columns:
        # Normalize the values between 0 and 1
        normalized_col = (cluster_gdf[col] - cluster_gdf[col].min()) / (
            cluster_gdf[col].max() - cluster_gdf[col].min()
        )
        # Avoid log(0) by replacing 0 with a small value
        normalized_col[normalized_col == 0] = 1e-10
        # Calculate probabilities for each value
        p = normalized_col / normalized_col.sum()
        # Calculate entropy for this column
        entropy_col = -np.sum(p * np.log(p)) / np.log(n)
        entropy[col] = entropy_col
    return pd.Series(entropy).sort_values(ascending=False)


def plot_flexibility_score(gdf, results, cluster_name):
    # Extract metrics' standard deviations (SDs)
    stats = results["basic_stats"]
    stds = stats["std"]  # Extract the 'std' column from the stats DataFrame
    sorted_stds = stds.sort_values()
    # Calculate entropy weights
    entropy_weights = entropy_weighting(gdf)
    sorted_entropy = entropy_weights.sort_values()
    # Normalize both SD and entropy for better combination
    normalized_stds = (sorted_stds - sorted_stds.min()) / (
        sorted_stds.max() - sorted_stds.min()
    )
    normalized_entropy = (sorted_entropy - sorted_entropy.min()) / (
        sorted_entropy.max() - sorted_entropy.min()
    )
    # Combine both normalized metrics to get a flexibility score
    flexibility_score = (
        normalized_stds + normalized_entropy
    ) / 2  # Averaging both for flexibility score
    # Plot the flexibility score
    plt.figure(figsize=(14, 8))
    bars = plt.bar(
        flexibility_score.index, flexibility_score, color="skyblue", width=0.8
    )
    # Find indices of the top 2 most flexible and bottom 2 strictest metrics
    top_2_flexible_indices = flexibility_score.nlargest(2).index
    bottom_2_strict_indices = flexibility_score.nsmallest(2).index
    # Highlight the top 2 flexible (green) and bottom 2 strict (red) metrics
    for idx in top_2_flexible_indices:
        bars[flexibility_score.index.get_loc(idx)].set_color("green")  # Top 2 flexible
    for idx in bottom_2_strict_indices:
        bars[flexibility_score.index.get_loc(idx)].set_color("red")  # Bottom 2 strict
    # Additionally, highlight all metrics with flexibility score close to 0 (threshold)
    threshold = 0.05
    close_to_zero_indices = flexibility_score[flexibility_score < threshold].index
    for idx in close_to_zero_indices:
        bars[flexibility_score.index.get_loc(idx)].set_color("red")
        # Add a minimum height to make them visible
        bar_height = max(flexibility_score[idx], 0.02)
        bars[flexibility_score.index.get_loc(idx)].set_height(bar_height)
        # Add text labels above the bars to show their actual values
        plt.text(
            flexibility_score.index.get_loc(idx),
            bar_height + 0.01,
            f"{flexibility_score[idx]:.0f}",
            ha="center",
            fontsize=10,
            color="black",
        )

    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha="right")
    # Add title and labels
    plt.title(f"Flexibility Score of Metrics for Cluster {cluster_name}", fontsize=16)
    plt.xlabel("Metrics")
    plt.ylabel("Flexibility Score (Avg of SD and Entropy)")
    # Show the plot
    plt.tight_layout()
    plt.show()


def unsupervised_leading_metrics(gdf):
    group_leading_metrics = {}
    # Standardize the features
    scaler = StandardScaler()
    features = gdf.drop(columns=["geometry"])
    # Standardize the data
    scaled_features = scaler.fit_transform(features)
    # Apply PCA
    pca = PCA(n_components=1)
    pca.fit(scaled_features)
    # Get the principal components
    components = pca.components_
    # Create a DataFrame of components with feature names
    leading_metrics = pd.DataFrame(components, columns=features.columns)
    # Sort metrics by their absolute contribution to the first component
    sorted_metrics = leading_metrics.iloc[0].abs().sort_values(ascending=False).head(5)
    group_leading_metrics["all_data"] = sorted_metrics
    return group_leading_metrics


def supervised_leading_metrics(gdf, classification_column):
    # Drop the classification and geometry columns for feature selection
    features = gdf.drop(columns=[classification_column, "geometry"])
    target = gdf[classification_column]

    # Standardize the features
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

    # Plot bar chart for feature importances
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

    return feature_importances


def cluster_similarity_analysis(gdf, classification_column):
    """
    Calculate and print silhouette and Davies-Bouldin scores for cluster similarity,
    and return silhouette values for plotting.
    """
    features = gdf.drop(columns=[classification_column, "geometry"])
    target = gdf[classification_column]

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Silhouette Scores
    silhouette_avg = silhouette_score(scaled_features, target)
    silhouette_values = silhouette_samples(scaled_features, target)

    # Davies-Bouldin Index
    db_score = davies_bouldin_score(scaled_features, target)

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Score: {db_score}")

    return silhouette_avg, silhouette_values, target


def classify_outliers(gdf, results):
    """
    Function to classify outliers in the global gdf based on the analysis results.

    Args:
        gdf: The main GeoDataFrame that contains all the data.
        classification_column: The column in gdf that represents the cluster classification.
        results: The results from the analyze_cluster function for each cluster.

    Returns:
        gdf: The main gdf with an added 'outlier_flag' column (1 for outliers, 0 for non-outliers).
    """
    # Initialize the outlier flag as 0 for all rows (assuming they are non-outliers)
    gdf["outlier_flag"] = 0

    # Loop through the results and identify the outliers for each cluster
    for cluster, cluster_result in results.items():
        # Extract the outliers from the result for this cluster
        outliers_in_cluster = cluster_result["outliers"]

        # Mark the outliers in the global gdf using the index from the cluster result
        gdf.loc[outliers_in_cluster.index, "outlier_flag"] = 1

    return gdf  # Return the updated gdf with the outlier_flag


def output_cluster_stats(cluster_label, results):
    """
    Function to save the cluster's basic statistics, flexibility scores, and other results into a CSV.
    Args:
        cluster_label: The name or label of the cluster.
        result: The analysis result containing stats, VIF, flexibility scores, etc.
        output_filename: Name of the CSV file to save the results.
    """
    # Extract relevant data from the result dictionary
    results = results[cluster_label]
    stats = results["basic_stats"]
    flexibility_score = results["flexibility_score"]
    vif_data = results["vif"]

    # Sort flexibility score in decreasing order
    sorted_flexibility = flexibility_score.sort_values(ascending=False)

    # Merge basic statistics with sorted flexibility score
    combined_df = stats.copy()
    combined_df["Flexibility Score"] = flexibility_score
    combined_df = combined_df.sort_values(by="Flexibility Score", ascending=False)

    # Add the VIF scores to the same DataFrame
    combined_df = combined_df.merge(
        vif_data.set_index("feature"), left_index=True, right_index=True, how="left"
    )
    combined_df["Cluster"] = cluster_label
    combined_df = combined_df.round(2)
    return combined_df


def varify_cleaned_gdf(gdf):
    threshold = len(gdf) * 0.5
    gdf = gdf.dropna(axis=1, thresh=threshold)
    gdf.fillna(0, inplace=True)
    gdf.drop(
        columns=["street_index", "junction_index", "tID", "tess_covered_area"],
        inplace=True,
    )
    return gdf


def analyze(gdf, csv_folder_path):
    gdf, results = analyze_gdf(gdf, "cluster", csv_folder_path)
    return gdf, results


if __name__ == "__main__":
    file_path = "../gpkg_files/modiin.gpkg"
    gdf = gpd.read_file(file_path)
    gdf = varify_cleaned_gdf(gdf)
    gdf, results = analyze_gdf(gdf, "cluster", "../output_CSVs")
    output_cluster_stats(3, results)
