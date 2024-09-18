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

    # Loop over each cluster in the classification column
    for cluster in gdf[classification_column].unique():
        cluster_rows = gdf[gdf[classification_column] == cluster]
        cluster_rows = cluster_rows.drop(
            columns=[classification_column]
        )  # Drop classification column
        cluster_results[cluster] = analyze_cluster(cluster_rows)

    # Global Analysis (Optional): Apply SVD over the whole dataset
    numeric_columns = gdf.select_dtypes(
        include=[float, int]
    ).columns  # Only numeric columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gdf[numeric_columns])

    # Perform SVD (using PCA for simplicity)
    svd = PCA(n_components=2)
    svd.fit(scaled_data)

    # Store global SVD components to compare across clusters
    global_svd_components = pd.DataFrame(svd.components_, columns=numeric_columns)

    # Compare clusters by their outlier ratios
    highest_outlier_cluster = max(
        cluster_results, key=lambda x: cluster_results[x]["outlier_ratio"]
    )
    lowest_outlier_cluster = min(
        cluster_results, key=lambda x: cluster_results[x]["outlier_ratio"]
    )

    print(
        f"Cluster with highest outlier ratio: {highest_outlier_cluster}, Ratio: {cluster_results[highest_outlier_cluster]['outlier_ratio']:.2f}"
    )
    print(
        f"Cluster with lowest outlier ratio: {lowest_outlier_cluster}, Ratio: {cluster_results[lowest_outlier_cluster]['outlier_ratio']:.2f}"
    )

    # Global summary to identify unique clusters and key metrics
    global_summary = {
        "svd_components": global_svd_components,
        "highest_outlier_cluster": highest_outlier_cluster,
        "lowest_outlier_cluster": lowest_outlier_cluster,
        "cluster_results": cluster_results,
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
    gdf["lof_score"] = lof.fit_predict(gdf[numeric_columns])
    lof_outliers = gdf[gdf["lof_score"] == -1]
    result["lof_outliers"] = lof_outliers  # Local outliers

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


def plot_gdf_analysis(gdf, results, cluster_number):
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


def plot_summary(results, cluster_name):
    # 8. Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Display number of outliers
    num_outliers = len(results["outliers"])
    num_lof_outliers = len(results["lof_outliers"])

    # Display the metrics with lower absolute loadings in PCA (more consistent)
    pca_components = results["pca_components"].T
    consistent_metrics = (
        pca_components.apply(lambda x: abs(x)).sum(axis=1).nsmallest(2).index.tolist()
    )

    # Correlation analysis summary (e.g., correlation matrix heatmap)
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

    ax.text(0.1, 0.5, summary_text, fontsize=12, ha="left", va="center", wrap=True)
    ax.set_axis_off()
    plt.title(f"Summary of Analysis for Cluster: {cluster_name}")
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


if __name__ == "__main__":
    data = {
        "building_height": [10, 12, 8, 15],
        "Convexity": [5, 9, 2, 2],
        "Compactness": [4, 33, 2, 11],
        "floor_aqrea": [33, 100, 200, 2],
        "orientation": [200, 250, 20, 10],
        "building_ERI": [100, 150, 120, 130],
        "building_area": [200, 250, 180, 300],
        "area_classification": ["1", "1", "2", "2"],
        "geometry": [Point(1, 2), Point(2, 3), Point(3, 4), Point(4, 5)],
    }

    gdf = gpd.GeoDataFrame(data, geometry="geometry")

    file_path = "../gpkg_files/simple_clustered_buildings.gpkg"

    # Print the full path
    # Load the GeoPackage as a GeoDataFrame
    gdf = gpd.read_file(file_path)
    classification_column = "area_classification"
    # leading_metrics_per_group = unsupervised_leading_metrics(gdf, classification_column, n_components=1)
    # top_metrics_per_group = supervised_leading_metrics(gdf, classification_column)
    gdf = gdf.drop(columns=["street_index"])
    gdf = gdf.dropna()
    cluster_number = 1
    cluster_1_rows = gdf[gdf["cluster"] == cluster_number]
    cluster_1_rows = cluster_1_rows.drop(columns=["cluster"])
    results = analyze_cluster(cluster_1_rows)

    plot_gdf_analysis(gdf, results, cluster_number)
    plot_summary(results, cluster_number)
