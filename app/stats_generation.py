import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bokeh.io import output_notebook
from bokeh.plotting import show
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


def analyze_gdf(gdf):
    result = {}

    # 1. Basic statistics: mean, variance, min, max
    stats = gdf.drop(columns=["geometry"]).describe().T
    stats["variance"] = gdf.drop(columns=["geometry"]).var()  # Variance
    result["basic_stats"] = stats

    # 2. Outlier detection using the IQR method (outliers are 1.5 * IQR beyond Q1 and Q3)
    Q1 = gdf.drop(columns=["geometry"]).quantile(0.25)
    Q3 = gdf.drop(columns=["geometry"]).quantile(0.75)
    IQR = Q3 - Q1
    outliers = (
        (gdf.drop(columns=["geometry"]) < (Q1 - 1.5 * IQR))
        | (gdf.drop(columns=["geometry"]) > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    result["outliers"] = gdf[outliers]  # Store rows that are outliers

    # 3. PCA to find leading metrics
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(gdf.drop(columns=["geometry"]))
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    result["pca_components"] = pd.DataFrame(
        pca.components_, columns=gdf.drop(columns=["geometry"]).columns
    )

    # 4. Correlation analysis between metrics
    correlation_matrix = gdf.drop(columns=["geometry"]).corr()
    result["correlation_matrix"] = correlation_matrix

    # 5. Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=20)
    gdf["lof_score"] = lof.fit_predict(gdf.drop(columns=["geometry"]))
    lof_outliers = gdf[gdf["lof_score"] == -1]
    result["lof_outliers"] = lof_outliers  # Local outliers

    return result


import geopandas as gpd


def plot_gdf_analysis(gdf, results):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Plot basic statistics (mean, variance)
    stats = results["basic_stats"]
    axes[0, 0].bar(
        stats.index, stats["mean"], yerr=stats["variance"], capsize=5, color="blue"
    )
    axes[0, 0].set_title("Mean and Variance of Metrics")
    axes[0, 0].set_xticklabels(stats.index, rotation=45, ha="right")

    # 2. Plot correlation matrix
    sns.heatmap(
        results["correlation_matrix"], annot=True, cmap="coolwarm", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Correlation Matrix of Metrics")

    # 3. Plot the outliers on a map using their geometry
    base = gdf.plot(ax=axes[1, 0], color="lightgrey")
    results["outliers"].plot(ax=base, color="red", marker="o", label="Outliers")
    axes[1, 0].set_title("Outliers Detected by IQR")

    # 4. Plot Local Outlier Factor (LOF) outliers on the map
    base = gdf.plot(ax=axes[1, 1], color="lightgrey")
    results["lof_outliers"].plot(
        ax=base, color="orange", marker="x", label="LOF Outliers"
    )
    axes[1, 1].set_title("Local Outlier Factor (LOF) Outliers")

    plt.tight_layout()
    plt.show()

    # Additional: Plot the PCA components in a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pca_df = results["pca_components"].T
    pca_df.plot(kind="bar", ax=ax)
    ax.set_title("Leading Metrics by PCA Components")
    ax.set_xticklabels(pca_df.index, rotation=45, ha="right")
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

    file_path = "../Notebooks/streets.gpkg"
    import os

    current_dir = os.path.dirname(__file__)

    # Construct the relative path to the file
    relative_path = "../Notebooks/streets.gpkg"

    # Combine the current directory with the relative path
    file_path = os.path.abspath(os.path.join(current_dir, relative_path))

    # Print the full path
    # Load the GeoPackage as a GeoDataFrame
    gdf = gpd.read_file(file_path)
    classification_column = "area_classification"
    # leading_metrics_per_group = unsupervised_leading_metrics(gdf, classification_column, n_components=1)
    # top_metrics_per_group = supervised_leading_metrics(gdf, classification_column)
    gdf = gdf.drop(columns=["from", "to"])
    print(gdf.isna().sum())
    print(gdf[gdf.isna().any(axis=1)])

    # print(gdf.head(5))
    # print(gdf.columns)
    analyze_gdf(gdf)
