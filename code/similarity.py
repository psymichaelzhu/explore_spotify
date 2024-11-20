
# %% Load data and packages
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.feature import StandardScaler as StandardScalerRDD
from pyspark.mllib.linalg.distributed import RowMatrix
import pyspark.sql.functions as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


spark = SparkSession \
        .builder \
        .appName("dr_cluster") \
        .getOrCreate()

# Read Spotify data
df = spark.read.csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True)


# Note potentially relevant features like danceability, energy, acousticness, etc.
df.columns

# %% column names
'''
['id',
 'name',
 'popularity',
 'duration_ms',
 'explicit', #ignore
 'release_date',
 'danceability',
 'energy',
 'key',
 'loudness',
 'mode',
 'speechiness',
 'acousticness',
 'instrumentalness',
 'liveness',#ignore
 'valence',
 'tempo',
 'time_signature',
 'artist']
 '''
# %% Data preprocessing
# identify potentially relevant features and add to a feature dataframe
feature_cols = [
    #'explicit',
    #'liveness',
    'duration_ms',
    'loudness',
    'key',
    'tempo', 
    'time_signature',
    'mode',
    'acousticness', 
    'instrumentalness',  
    'speechiness',
    'danceability', 
    'energy',
    'valence'
    ]

# select feature columns and numeric data as floats
df_features = df.select(*(F.col(c).cast("float").alias(c) for c in feature_cols),'id','name', 'artist') \
                         .dropna()
df_features = df_features.withColumn('features', F.array(*[F.col(c) for c in feature_cols])) \
                         .select('id','name', 'artist', 'features')

# convert features to dense vector format (expected by K-Means, PCA)
vectors = df_features.rdd.map(lambda row: Vectors.dense(row.features))
features = spark.createDataFrame(vectors.map(Row), ["features_unscaled"])

# scale features (some values like duration_ms are much larger than others)
standardizer = StandardScaler(inputCol="features_unscaled", outputCol="features")
model = standardizer.fit(features)
features = model.transform(features) \
                .select('features')

# persist in memory before fit model
features.persist()
features.printSchema()


# %% K-means
# try different numbers of clusters to find optimal k
def find_optimal_kmeans(features, k_values=range(2, 6)):
    """
    Find optimal k for KMeans clustering using silhouette scores
    
    Args:
        features: DataFrame with feature vectors
        k_values: Range of k values to try
        
    Returns:
        optimal_k: Optimal number of clusters
        kmeans_predictions: Predictions using optimal k
        silhouettes: List of silhouette scores for each k
    """
    silhouettes = []
    
    for k in k_values:
        # train model
        kmeans = KMeans(k=k, seed=1)
        model = kmeans.fit(features)
        
        # make predictions
        predictions = model.transform(features)
        
        # evaluate clustering
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        silhouettes.append(silhouette)
        print(f"Silhouette score for k={k}: {silhouette}")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouettes, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    
    # Add value labels on the points
    for k, silhouette in zip(k_values, silhouettes):
        plt.annotate(f'{silhouette:.3f}', 
                    (k, silhouette), 
                    textcoords="offset points", 
                    xytext=(0,10),
                    ha='center')
    
    plt.show()

    # find optimal k
    optimal_k = k_values[silhouettes.index(max(silhouettes))]
    print(f"\nOptimal number of clusters (k) = {optimal_k}")
    print(f"Best silhouette score = {max(silhouettes)}")
    
    # train final model with optimal k
    kmeans = KMeans(k=optimal_k, seed=1)
    kmeans_model = kmeans.fit(features)
    optimal_predictions = kmeans_model.transform(features)
    
    return optimal_k, optimal_predictions, silhouettes

# %% PCA-KMeans
def find_optimal_pca_components(features,threshold=0.9,k=None):
    """
    Find optimal number of PCA components by analyzing explained variance
    
    Args:
        features: DataFrame with feature vectors
        feature_cols: List of feature column names
        
    Returns:
        explained_variances: Array of explained variance ratios
        cumulative_variance: Array of cumulative explained variance ratios
    """
    n_features = features.first()["features"].size
    pca = PCA(k=n_features, inputCol="features", outputCol="pcaFeatures") 
    model = pca.fit(features)

    explained_variances = model.explainedVariance
    cumulative_variance = [sum(explained_variances[:i+1]) for i in range(len(explained_variances))]

    # Print variance explained by each component
    for i, var in enumerate(explained_variances):
        print(f"Component {i+1} explained variance: {var:.4f}")
        print(f"Cumulative explained variance: {cumulative_variance[i]:.4f}")

    # Find optimal number of components
    if k is None:
        optimal_n = next((i for i, cum_var in enumerate(cumulative_variance) if cum_var >= threshold), len(cumulative_variance))
    else:
        optimal_n = k

    # Plot explained variance analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot scree plot
    ax1.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal number of components ({optimal_n})')

    # Plot cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    # Add threshold line at 0.8
    if k is None:
        ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax2.legend()
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance Plot')

    plt.tight_layout()
    plt.show()

    if k is None:
        print(f"Optimal number of components: {optimal_n}, threshold: {threshold}")
    else:
        print(f"Optimal number of components: {optimal_n}")

    # transform features according to optimal number of components
    pca = PCA(k=optimal_n, inputCol="features", outputCol="pcaFeatures") 
    model = pca.fit(features)

    pca_results = model.transform(features) \
                        .select('pcaFeatures')
    
    pca_features = pca_results.rdd.map(lambda row: Vectors.dense(row.pcaFeatures))
    pca_features = spark.createDataFrame(pca_features.map(Row), ["features"])

    return optimal_n, pca_features, explained_variances, cumulative_variance
# 1. PCA: find optimal number of components
optimal_n, features_pca, explained_variances, cumulative_variance = find_optimal_pca_components(features,k=2)
# 2. KMeans: find optimal k, based on PCA-transformed features
features_pca.persist()
optimal_k_pca, kmeans_predictions_pca, silhouettes_pca = find_optimal_kmeans(features_pca)
#examine cluster distribution
kmeans_predictions_pca.groupby('prediction') \
               .count() \
               .show()

# %% visualize clusters for a specific artist

def plot_cluster_radar(predictions_df, original_df, artist_name, feature_cols, num_clusters):
    """
    Create a radar plot to compare cluster centroids for a specific artist with standardized features
    """
    # Merge predictions with original features
    df_merged = predictions_df.withColumn("id", F.monotonically_increasing_id()) \
            .join(original_df.withColumn("id", F.monotonically_increasing_id())
                 .withColumnRenamed("features", "raw_features"), on="id", how="inner")
    
    # Calculate global means and standard deviations for standardization
    global_stats = df_merged.select([
        *[F.avg(F.expr(f"raw_features[{i}]")).alias(f"{col}_mean") for i, col in enumerate(feature_cols)],
        *[F.stddev(F.expr(f"raw_features[{i}]")).alias(f"{col}_std") for i, col in enumerate(feature_cols)]
    ]).collect()[0]
    
    # Filter for specific artist
    artist_data = df_merged.filter(F.col("artist") == artist_name)
    
    # Calculate cluster means and standardize them
    cluster_means = artist_data.groupBy("prediction").agg(*[
        F.avg(F.expr(f"raw_features[{i}]")).alias(col)
        for i, col in enumerate(feature_cols)
    ])
    
    # Convert to pandas for plotting
    cluster_means_pd = cluster_means.toPandas()
    
    # Ensure all clusters are represented
    all_clusters = pd.DataFrame({'prediction': range(num_clusters)})
    cluster_means_pd = pd.merge(all_clusters, cluster_means_pd, on='prediction', how='left')
    
    # Standardize the means using global statistics
    for i, col in enumerate(feature_cols):
        mean_val = global_stats[f"{col}_mean"]
        std_val = global_stats[f"{col}_std"]
        if std_val > 0:  # Avoid division by zero
            cluster_means_pd[col] = (cluster_means_pd[col] - mean_val) / std_val
    
    # Set up the angles for
    angles = np.linspace(0, 2*np.pi, len(feature_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the circle
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Define colors
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    
    # Plot data for each cluster
    for cluster in range(num_clusters):
        values = cluster_means_pd[cluster_means_pd['prediction'] == cluster][feature_cols].values
        if len(values) > 0 and not np.isnan(values).all():
            values = values[0]
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', 
                   color=colors[cluster])
            ax.fill(angles, values, alpha=0.25, color=colors[cluster])
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=8)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(f"Cluster Characteristics for {artist_name}")
    plt.tight_layout()
    plt.show()

def plot_cluster_distribution(artist_data, num_clusters):
    """Plot distribution of songs across clusters"""
    # Define consistent colors
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    
    # Ensure all clusters are represented
    all_clusters = pd.Series(0, index=range(num_clusters))
    cluster_counts = artist_data['prediction'].value_counts()
    cluster_counts = cluster_counts.combine_first(all_clusters)
    cluster_counts = cluster_counts.sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cluster_counts.index, cluster_counts.values)
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
        
    plt.xlabel('Cluster')
    plt.ylabel('Number of Songs')
    plt.title(f'Distribution of Songs Across Clusters')
    plt.show()
    
    return cluster_counts

def plot_cluster_evolution(artist_data, num_clusters):
    """Plot evolution of clusters over time"""
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    
    # Calculate proportion of each cluster per year
    yearly_proportions = artist_data.pivot_table(
        index='year',
        columns='prediction',
        aggfunc='size',
        fill_value=0
    )
    yearly_proportions = yearly_proportions.div(yearly_proportions.sum(axis=1), axis=0)
    
    plt.figure(figsize=(10, 6))
    yearly_proportions.plot(
        kind='area',
        stacked=True,
        alpha=0.7,
        color=[colors[i] for i in range(num_clusters)]
    )
    plt.xlabel('Year')
    plt.ylabel('Proportion of Songs')
    plt.title('Evolution of Song Clusters Over Time')
    plt.show()

def plot_cluster_scatter(df_merged, artist_name, num_clusters, dim_1, dim_2):
    """Plot scatter of clusters in PCA space"""
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h']
    
    plt.figure(figsize=(10, 8))
    
    artist_matrix = df_merged.filter(F.col("artist") == artist_name) \
                          .select("name", "prediction", "features") \
                          .toPandas()

    for cluster in artist_matrix["prediction"].unique():
        mask = artist_matrix["prediction"] == cluster
        plt.scatter(artist_matrix[mask]["features"].apply(lambda x: float(x[dim_1])),
                   artist_matrix[mask]["features"].apply(lambda x: float(x[dim_2])),
                   c=[colors[cluster]],
                   marker=markers[cluster % len(markers)],
                   label=f'Cluster {cluster}',
                   s=100)

    for i, txt in enumerate(artist_matrix["name"]):
        x = float(artist_matrix["features"].iloc[i][dim_1])
        y = float(artist_matrix["features"].iloc[i][dim_2])
        plt.annotate(txt, (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8)

    plt.xlabel(f"Component {dim_1}")
    plt.ylabel(f"Component {dim_2}")
    plt.title(f"Songs Clustered in Dimensionally Reduced Space")
    plt.legend()
    plt.show()

def plot_feature_distributions(df_features, kmeans_predictions_pca, artist_name, feature_cols, num_clusters):
    """Plot distribution of original features for each cluster"""
    plt.figure(figsize=(15, 10))

    artist_features = df_features.filter(F.col("artist") == artist_name) \
        .withColumn("idx", F.monotonically_increasing_id())

    predictions_with_idx = kmeans_predictions_pca \
        .withColumn("idx", F.monotonically_increasing_id())

    artist_features_with_clusters = artist_features.join(
        predictions_with_idx.select("idx", "prediction"), 
        on="idx"
    ).select(
        "prediction",
        *[F.expr(f"features[{pos}]").alias(col) for pos, col in enumerate(feature_cols)]
    )

    features_pd = artist_features_with_clusters.toPandas()

    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()

    if len(features_pd) > 0:
        existing_clusters = features_pd['prediction'].unique()
        missing_clusters = [c for c in range(num_clusters) if c not in existing_clusters]
        
        if missing_clusters:
            dummy_data = pd.DataFrame({
                'prediction': missing_clusters,
                **{col: [float('nan')] * len(missing_clusters) for col in feature_cols}
            })
            features_pd = pd.concat([features_pd, dummy_data], ignore_index=True)

    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        features_pd.boxplot(column=feature, by='prediction', ax=ax)
        ax.set_title(f'{feature} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_xticklabels(range(num_clusters))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution of Features Across Clusters', y=1.02)
    plt.tight_layout()
    plt.show()

def analyze_artist_clusters(predictions_df, original_df, artist_name, num_clusters, dim_1, dim_2):
    """
    Analyze and visualize clustering results for a specific artist
    
    Args:
        predictions_df: DataFrame with clustering predictions
        original_df: Original DataFrame with all song information
        artist_name: Name of the artist to analyze
        dim_1: index of the first dimension to plot
        dim_2: index of the second dimension to plot
    """
    
               
    df_merged = predictions_df.withColumn("id", F.monotonically_increasing_id()) \
            .join(original_df.withColumn("id", F.monotonically_increasing_id()).withColumnRenamed("features", "raw_features"), on="id", how="inner") 
    
    results = df_merged.join(
        df.select('release_date', 'name', 'artist'), 
        on=['name', 'artist'], 
        how='left'
    ).filter(F.col('artist') == artist_name)
        
    artist_data = results.toPandas()
    artist_data['release_date'] = pd.to_datetime(artist_data['release_date'])
    artist_data['year'] = artist_data['release_date'].dt.year

    # Plot all visualizations using the helper functions
    cluster_counts = plot_cluster_distribution(artist_data, num_clusters)
    plot_cluster_evolution(artist_data, num_clusters)
    plot_cluster_scatter(df_merged, artist_name, num_clusters, dim_1, dim_2)
    plot_cluster_radar(predictions_df, original_df, artist_name, feature_cols, num_clusters)
    #plot_feature_distributions(df_features, predictions_df, artist_name, feature_cols, num_clusters)

    # Print summary statistics
    print(f"\nSummary for {artist_name}:")
    print(f"Total number of songs: {len(artist_data)}")
    print("\nSongs per cluster:")
    print(cluster_counts)
    
    return artist_data

#demo
artist_name="Coldplay"
analyze_artist_clusters(kmeans_predictions_pca, df_features, artist_name,num_clusters=optimal_k_pca,dim_1=0,dim_2=1)

#%%
def plot_global_cluster_radar(predictions_df, original_df, feature_cols, num_clusters):
    """
    Create a radar plot to compare cluster centroids for all songs with standardized features
    """
    # Merge predictions with original features
    df_merged = predictions_df.withColumn("id", F.monotonically_increasing_id()) \
            .join(original_df.withColumn("id", F.monotonically_increasing_id())
                 .withColumnRenamed("features", "raw_features"), on="id", how="inner")
    
    # Sample a subset of the data for visualization
    df_merged = df_merged.sample(False, 0.1, seed=42)  # Sample 10% of data randomly
    df_merged.show()
    # Calculate global means and standard deviations for standardization
    global_stats = df_merged.select([
        *[F.avg(F.expr(f"raw_features[{i}]")).alias(f"{col}_mean") for i, col in enumerate(feature_cols)],
        *[F.stddev(F.expr(f"raw_features[{i}]")).alias(f"{col}_std") for i, col in enumerate(feature_cols)]
    ]).collect()[0]
    
    # Calculate cluster means
    cluster_means = df_merged.groupBy("prediction").agg(*[
        F.avg(F.expr(f"raw_features[{i}]")).alias(col)
        for i, col in enumerate(feature_cols)
    ])
    
    # Convert to pandas for plotting
    cluster_means_pd = cluster_means.toPandas()
    
    # Ensure all clusters are represented
    all_clusters = pd.DataFrame({'prediction': range(num_clusters)})
    cluster_means_pd = pd.merge(all_clusters, cluster_means_pd, on='prediction', how='left')
    
    # Standardize the means using global statistics
    for i, col in enumerate(feature_cols):
        mean_val = global_stats[f"{col}_mean"]
        std_val = global_stats[f"{col}_std"]
        if std_val > 0:  # Avoid division by zero
            cluster_means_pd[col] = (cluster_means_pd[col] - mean_val) / std_val
    
    # Set up the angles
    angles = np.linspace(0, 2*np.pi, len(feature_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the circle
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Define colors
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    
    # Plot data for each cluster
    for cluster in range(num_clusters):
        if cluster == 3:
            continue
        values = cluster_means_pd[cluster_means_pd['prediction'] == cluster][feature_cols].values
        if len(values) > 0 and not np.isnan(values).all():
            values = values[0]
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', 
                   color=colors[cluster])
            ax.fill(angles, values, alpha=0.25, color=colors[cluster])
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=10)
    
    # Add gridlines and adjust their style
    ax.grid(True, alpha=0.3)
    
    # Add legend with a better position
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title("Global Cluster Characteristics", pad=20, size=14)
    
    plt.tight_layout()
    plt.show()


plot_global_cluster_radar(kmeans_predictions_pca, df_features, feature_cols, optimal_k_pca)

# %%
