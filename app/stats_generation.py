import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import pandas as pd
from esda.moran import Moran
import libpysal
from esda.getisord import G
from sklearn.cluster import KMeans
import geopandas as gpd
from scipy.stats import skew, kurtosis, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import entropy
import shap
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

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


def analyze_gdf(gdf, classification_column):
    cluster_results = {}

    # print(gdf["lof_score"].head(10))
    # Loop over each cluster in the classification column
    for cluster in gdf[classification_column].unique():
        cluster_rows = gdf[gdf[classification_column] == cluster]
        cluster_rows = cluster_rows.drop(
            columns=[classification_column]
        )  # Drop classification column

        numeric_data = cluster_rows.select_dtypes(
            include=[float, int]
        )  # Select only numeric columns
        geometry_data = cluster_rows["geometry"]  # Keep the geometry column separate

        # Standardize the numeric columns
        scaler = StandardScaler()
        numeric_data_standardized = pd.DataFrame(
            scaler.fit_transform(numeric_data), columns=numeric_data.columns
        )

        # Re-attach the geometry column to the standardized numeric data
        cluster_rows = gpd.GeoDataFrame(
            numeric_data_standardized, geometry=geometry_data
        )
        cluster_results[cluster] = analyze_cluster(cluster_rows)
        print("cluster", cluster)
        if cluster == 7:
            plot_cluster_summary(cluster_rows, cluster_results[cluster], cluster)
            plot_cluster_analysis(cluster_rows, cluster_results[cluster], cluster)
        plot_flexibility_score(cluster_rows, cluster_results[cluster], cluster)
    # Global Analysis (Optional): Apply SVD over the whole dataset
    numeric_columns = gdf.select_dtypes(include=[float, int]).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gdf[numeric_columns])

    # Perform SVD (using PCA for simplicity)
    svd = PCA(n_components=2)
    svd.fit(scaled_data)

    # Store global SVD components to compare across clusters
    global_svd_components = pd.DataFrame(svd.components_, columns=numeric_columns)

    # Supervised analysis to find the leading metrics for classification
    supervised_importances = supervised_leading_metrics(gdf, classification_column)

    # Cluster similarity analysis
    similarity_results = cluster_similarity_analysis(gdf, classification_column)
    print(similarity_results)
    # Compare clusters by their outlier ratios
    highest_outlier_cluster = max(
        cluster_results, key=lambda x: cluster_results[x]["outlier_ratio"]
    )
    lowest_outlier_cluster = min(
        cluster_results, key=lambda x: cluster_results[x]["outlier_ratio"]
    )

    # Closest clusters based on centroid distances
    # closest_clusters = similarity_results["closest_clusters"]
    # cluster_names = gdf[classification_column].unique()

    # closest_cluster_1 = cluster_names[closest_clusters[0]]
    # closest_cluster_2 = cluster_names[closest_clusters[1]]

    print(
        f"Cluster with highest outlier ratio: {highest_outlier_cluster}, Ratio: {cluster_results[highest_outlier_cluster]['outlier_ratio']:.2f}"
    )
    print(
        f"Cluster with lowest outlier ratio: {lowest_outlier_cluster}, Ratio: {cluster_results[lowest_outlier_cluster]['outlier_ratio']:.2f}"
    )
    # print(f"Closest clusters: {closest_cluster_1} and {closest_cluster_2}")

    # Global summary to identify unique clusters and key metrics
    global_summary = {
        "svd_components": global_svd_components,
        "supervised_leading_metrics": supervised_importances,
        "highest_outlier_cluster": highest_outlier_cluster,
        "lowest_outlier_cluster": lowest_outlier_cluster,
        # "cluster_results": cluster_results,
        # "silhouette_avg": similarity_results["silhouette_avg"],
        # "db_score": similarity_results["db_score"],
        # "closest_clusters": (closest_cluster_1, closest_cluster_2),
        # "cluster_distance_matrix": similarity_results["cluster_distance_matrix"],
    }

    return global_summary


def analyze_cluster(gdf):
    result = {}
    # 1. Basic statistics: mean, variance, min, max
    numeric_columns = gdf.select_dtypes(
        include=[float, int]
    ).columns  # Only numeric columns
    stats = gdf[numeric_columns].describe().T
    stats["variance"] = gdf[numeric_columns].var()  # Variance
    result["basic_stats"] = stats

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
    result["metrics_influence"] = metrics_influence.sum().sort_values(
        ascending=False
    )  # Summing up outlier contributions

    # 4. PCA to find leading metrics
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gdf[numeric_columns])
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    result["pca_components"] = pd.DataFrame(pca.components_, columns=numeric_columns)

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
    result["lof_outliers"]= lof_outliers

    # 7. Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = (
        numeric_columns  # Ensure these are the same columns as scaled_data
    )
    vif_data["VIF"] = [
        variance_inflation_factor(scaled_data, i) for i in range(scaled_data.shape[1])
    ]
    result["vif"] = vif_data

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

    # 6. Plot PCA components in a separate figure
    fig, ax = plt.subplots(figsize=(10, 6))
    pca_df = results["pca_components"].T
    pca_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Leading Metrics by PCA Components for Cluster {cluster_number}")
    ax.set_xticklabels(pca_df.index, rotation=45, ha="right")
    plt.show()

    # 7. Plot the metrics that influenced outliers the most
    fig, ax = plt.subplots(figsize=(10, 6))
    results["metrics_influence"].plot(kind="bar", ax=ax, color="green")
    ax.set_title(
        f"Metrics That Influenced Outliers the Most for Cluster {cluster_number}"
    )
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_cluster_summary(gdf, results, cluster_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display number of outliers
    num_outliers = len(results["outliers"])
    num_lof_outliers = len(results["lof_outliers"])

    # Display the metrics with lower absolute loadings in PCA (more consistent)
    pca_components = results["pca_components"].T
    consistent_metrics = (
        pca_components.apply(lambda x: abs(x)).sum(axis=1).nsmallest(2).index.tolist()
    )

    # Correlation analysis summary
    corr_summary = results["correlation_matrix"].abs().mean().mean()

    # Extract metrics with the highest and lowest SD
    stats = results["basic_stats"]
    highest_sd_metrics = stats["std"].nlargest(2).index.tolist()  # Highest SD
    lowest_sd_metrics = stats["std"].nsmallest(2).index.tolist()  # Lowest SD

    # Extract metrics with the highest and lowest mean
    highest_mean_metrics = stats["mean"].nlargest(2).index.tolist()  # Highest Mean
    lowest_mean_metrics = stats["mean"].nsmallest(2).index.tolist()  # Lowest Mean

    # Create the summary text
    summary_text = (
        f"Cluster: {cluster_name}\n"
        f"Number of IQR outliers: {num_outliers}\n"
        f"Number of LOF outliers: {num_lof_outliers}\n"
        f"More Consistent Metrics from PCA: {', '.join(consistent_metrics)}\n"
        f"Average Correlation: {corr_summary:.2f}\n"
        f"Metrics Influencing Outliers: {', '.join(results['metrics_influence'].index[:3])}\n"
        f"Metrics with Highest SD: {', '.join(highest_sd_metrics)}\n"
        f"Metrics with Lowest SD: {', '.join(lowest_sd_metrics)}\n"
        f"Metrics with Highest Mean: {', '.join(highest_mean_metrics)}\n"
        f"Metrics with Lowest Mean: {', '.join(lowest_mean_metrics)}"
    )

    # Improve the text display on the plot
    ax.text(0.1, 0.5, summary_text, fontsize=12, ha="left", va="center", wrap=True)
    ax.set_axis_off()
    plt.title(f"Summary of Analysis for Cluster: {cluster_name}", fontsize=16)
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
    features = gdf.drop(columns=[classification_column, "geometry"])
    target = gdf[classification_column]
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # Silhouette Score
    silhouette_avg = silhouette_score(scaled_features, target)
    # Davies-Bouldin Index
    db_score = davies_bouldin_score(scaled_features, target)
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Score: {db_score}")

    return silhouette_avg, db_score


if __name__ == "__main__":
    file_path = "../gpkg_files/modiin.gpkg"
    gdf = gpd.read_file(file_path)
    # gdf.drop(columns="geometry").to_csv("output.csv", index=False)
    threshold = len(gdf) * 0.5

    # Step 1: Remove columns with more than 50% NaNs
    gdf = gdf.dropna(axis=1, thresh=threshold)

    # Step 2: Replace remaining NaNs with 0 in the remaining columns
    gdf.fillna(0, inplace=True)

    # Output the cleaned GeoDataFrame
    print("Columns removed due to more than 50% NaNs:")
    print(set(gdf.columns) - set(gdf.columns))
    # Optionally, print the cleaned GeoDataFrame if needed
    gdf.drop(columns=["street_index", "junction_index", "tID"], inplace=True)
    # cluster = gdf["cluster"]
    # gdf = gdf.drop(columns=["cluster"])
    analyze_gdf(gdf, "cluster")

    # data = {
    #     "building_height": [10, 12, 8, 15],
    #     "Convexity": [5, 9, 2, 2],
    #     "Compactness": [4, 33, 2, 11],
    #     "floor_aqrea": [33, 100, 200, 2],
    #     "orientation": [200, 250, 20, 10],
    #     "building_ERI": [100, 150, 120, 130],
    #     "building_area": [200, 250, 180, 300],
    #     "area_classification": ["1", "1", "2", "2"],
    #     "geometry": [Point(1, 2), Point(2, 3), Point(3, 4), Point(4, 5)],
    # }

    # gdf = gpd.GeoDataFrame(data, geometry="geometry")

    # file_path = "../gpkg_files/modiin.gpkg"

    # gdf = gpd.read_file(file_path)
    # # print(gdf.head(10))
    # # leading_metrics_per_group = unsupervised_leading_metrics(gdf, classification_column, n_components=1)
    # # top_metrics_per_group = supervised_leading_metrics(gdf, classification_column)
    # # gdf = gdf.drop(columns=["street_index"])
    # gdf = gdf.dropna()
    # cluster_number = 1
    # cluster_1_rows = gdf[gdf["cluster"] == cluster_number]
    # print(gdf['cluster'].unique())
    # # print(cluster_1_rows.head(10))
    # cluster_1_rows = cluster_1_rows.drop(columns=["cluster"])
    # results = analyze_cluster(cluster_1_rows)

    # plot_cluster_analysis(gdf, results, cluster_number)
    # plot_cluster_summary(gdf, results, cluster_number)

    # example 2
    # file_path = "../gpkg_files/katzrin.gpkg"
    # gdf = gpd.read_file(file_path)
    # gdf = gdf.dropna()
    # columns = gdf.columns
    # results = analyze_cluster(gdf)
    # plot_gdf_analysis(gdf, results, 1)
    # plot_cluster_summary(gdf, results, 1)
