import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_and_clean_data(filepath, year1, year2):
    """
    Load and clean GDP data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.
    year1 (str): The first year for GDP data column.
    year2 (str): The second year for GDP data column.

    Returns:
    DataFrame: A cleaned DataFrame with selected years and non-null values.
    """
    gdp_data = pd.read_csv(filepath, skiprows=4)
    gdp_data = gdp_data[['Country Name', year1, year2]].dropna()
    return gdp_data


def normalize_data(data):
    """
    Normalize the data using Standard Scaler.

    Parameters:
    data (DataFrame): DataFrame to be normalized.

    Returns:
    ndarray: Normalized data.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def find_optimal_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters for KMeans clustering using 
    silhouette scores.

    Parameters:
    data (ndarray): The data to cluster.
    max_clusters (int): The maximum number of clusters to test.

    Returns:
    tuple: A tuple containing the optimal number of clusters and a list of 
    silhouette scores.
    """
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    optimal_clusters = np.argmax(silhouette_scores) + 2
    return optimal_clusters, silhouette_scores


def plot_clusters_with_centers(data, labels, centers, year1, year2):
    """
    Plot the clusters along with their centers.

    Parameters:
    data (ndarray): The clustered data.
    labels (ndarray): Cluster labels for each data point.
    centers (ndarray): Coordinates of the cluster centers.
    year1 (str): The first year for GDP data column.
    year2 (str): The second year for GDP data column.
    """
    marker_styles = ['o', 's', '^']
    plt.figure(figsize=(10, 8))
    for cluster, marker in zip(range(np.max(labels) + 1), marker_styles):
        cluster_data = data[labels == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=100,
                    marker=marker, label=f'Cluster {cluster}', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], marker='*',
                color='black', s=300, label='Centers')
    plt.title(
        f'Clustering of Countries by GDP per Capita ({year1} vs {year2})',
        fontsize=18,fontweight='bold')
    plt.xlabel(f'GDP per Capita in {year2}', fontsize=16)
    plt.ylabel(f'GDP per Capita in {year1}', fontsize=1)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# Main execution block
if __name__ == '__main__':
    # Load and clean data
    filepath = 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv'  # File path
    year1, year2 = '1990', '2020'
    gdp_data = load_and_clean_data(filepath, year1, year2)

    # Normalize data
    normalized_data = normalize_data(gdp_data[[year1, year2]])

    # Find the optimal number of clusters
    optimal_clusters, silhouette_scores = find_optimal_clusters(
        normalized_data)

    # Apply K-Means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(normalized_data)

    # Plot clusters with centers
    plot_clusters_with_centers(
        normalized_data, cluster_labels, kmeans.cluster_centers_, year1, year2)

    # Load the full GDP data
    gdp_data = pd.read_csv(filepath, skiprows=4)

   
