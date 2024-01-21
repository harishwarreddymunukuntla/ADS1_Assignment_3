import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import errors as err


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
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=120,
                    marker=marker, label=f'Cluster {cluster}', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], marker='*',
                color='black', s=300, label='Centers')
    plt.title(
        f'Clustering of Countries by GDP per Capita ({year1} vs {year2})',
        fontsize=18,fontweight='bold')
    plt.xlabel(f'GDP per Capita in {year2}', fontsize=16)
    plt.ylabel(f'GDP per Capita in {year1}', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


def exponential_growth(x, a, b, c):
    """
    Exponential growth model.

    Parameters:
    x (ndarray): Independent variable.
    a, b, c (float): Parameters of the model.

    Returns:
    ndarray: Calculated values of the dependent variable.
    """
    return a * np.exp(b * x) + c


def extract_gdp_data(gdp_data, country_name):
    """
    Extract and process GDP data for a specific country.

    Parameters:
    gdp_data (DataFrame): Full GDP data.
    country_name (str): Name of the country.

    Returns:
    DataFrame: Processed GDP data for the specified country.
    """
    country_gdp_df = gdp_data[gdp_data['Country Name'] == country_name]
    country_gdp_values = country_gdp_df.iloc[0, 4:-1].values
    years = country_gdp_df.columns[4:-1]
    country_gdp_df_cleaned = pd.DataFrame(
        {'Year': years, 'GDP_per_capita': country_gdp_values})
    country_gdp_df_cleaned['Year'] = country_gdp_df_cleaned['Year'].astype(int)
    country_gdp_df_cleaned['GDP_per_capita'] = pd.to_numeric(
        country_gdp_df_cleaned['GDP_per_capita'], errors='coerce')
    country_gdp_df_cleaned.dropna(inplace=True)
    return country_gdp_df_cleaned


def fit_and_predict(gdp_df):
    """
    Fit the exponential growth model to the GDP data and make future 
    predictions.

    Parameters:
    gdp_df (DataFrame): GDP data of a specific country.

    Returns:
    tuple: Model parameters, covariance, start year, future years, 
    predictions, and error ranges.
    """
    x_data = gdp_df['Year'] - gdp_df['Year'].min()
    y_data = gdp_df['GDP_per_capita']
    
    # Adjusting initial parameter values for a better fit
    initial_params = [y_data.min(), 0.1, y_data.min()]
    
    params, covar = curve_fit(
        exponential_growth, x_data, y_data, p0=initial_params)
    
    # Limiting the prediction range to 10 years into the future
    future_years = np.array([40])
    future_x_data = np.max(x_data) + future_years
    future_predictions = exponential_growth(future_x_data, *params)
    error_ranges = err.error_prop(
        future_x_data, exponential_growth, params, covar)
    return params, covar, gdp_df['Year'].min(), future_x_data, \
        future_predictions, error_ranges


def plot_gdp_data(gdp_df, params, covar, start_year, future_x_data,
                  future_predictions, error_ranges, country_name):
    """
    Plot the GDP data along with the model fit and future predictions.

    Parameters:
    gdp_df (DataFrame): GDP data of a specific country.
    params (ndarray): Parameters of the fitted model.
    covar (ndarray): Covariance matrix of the fitted model.
    start_year (int): Starting year of the GDP data.
    future_x_data (ndarray): Future years for prediction.
    future_predictions (ndarray): Predicted values for future years.
    error_ranges (ndarray): Error ranges for the predictions.
    country_name (str): Name of the country.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(gdp_df['Year'], gdp_df['GDP_per_capita'],
                color='blue', label=f'Actual GDP Data ({country_name})')
    years_extended = np.linspace(
        gdp_df['Year'].min() - start_year, future_x_data.max(), 100)
    gdp_fitted = exponential_growth(years_extended, *params)
    plt.plot(start_year + years_extended, gdp_fitted,
             color='red', label='Fitted Exponential Model')
    error_extended = err.error_prop(
        years_extended, exponential_growth, params, covar)
    lower_bounds_extended = gdp_fitted - error_extended
    upper_bounds_extended = gdp_fitted + error_extended
    plt.fill_between(start_year + years_extended, lower_bounds_extended,
                     upper_bounds_extended, color='green', alpha=0.3,
                     label='Confidence Interval')
    plt.scatter(start_year + future_x_data, future_predictions,
                color='black', marker='x', label='Predictions')
    plt.xlabel('Year',fontsize=16)
    plt.ylabel('GDP Per Capita (USD)', fontsize=16)
    plt.title(
        f'GDP Per Capita Over Time for {country_name}',fontsize=18, 
        fontweight='bold')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.yscale('log')
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

    # Extract and process data for India and United Kingdom
    india_gdp_df = extract_gdp_data(gdp_data, 'India')
    uk_gdp_df = extract_gdp_data(gdp_data, 'United Kingdom')
    Sweden_gdp_df = extract_gdp_data(gdp_data, 'Sweden')

    # Fit model and make predictions for India
    india_params, india_covar, india_start_year, india_future_x_data,\
        india_future_predictions, india_error_ranges = fit_and_predict(
            india_gdp_df)

    # Fit model and make predictions for United Kingdom
    uk_params, uk_covar, uk_start_year, uk_future_x_data, \
        uk_future_predictions, uk_error_ranges = fit_and_predict(
            uk_gdp_df)

    # Fit model and make predictions for Sweden
    Sweden_params, Sweden_covar, Sweden_start_year, Sweden_future_x_data, \
        Sweden_future_predictions, Sweden_error_ranges = fit_and_predict(
            Sweden_gdp_df)

    # Plotting for India
    plot_gdp_data(india_gdp_df, india_params, india_covar,
                  india_start_year,
                  india_future_x_data, india_future_predictions,
                  india_error_ranges, 'India')

    # Plotting for United Kingdom
    plot_gdp_data(uk_gdp_df, uk_params, uk_covar, uk_start_year,
                  uk_future_x_data,
                  uk_future_predictions, uk_error_ranges, 'United Kingdom')

    # Plotting for Sweden
    plot_gdp_data(Sweden_gdp_df, Sweden_params, Sweden_covar,
                  Sweden_start_year, Sweden_future_x_data,
                  Sweden_future_predictions, Sweden_error_ranges, 'Sweden')
