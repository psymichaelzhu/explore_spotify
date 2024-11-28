# %% Load data and packages
#environment: amd 2nodes 60G 
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.feature import StandardScaler as StandardScalerRDD
from pyspark.mllib.linalg.distributed import RowMatrix
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType, ArrayType
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from prophet import Prophet
from scipy.stats import gaussian_kde


spark = SparkSession \
        .builder \
        .appName("dr_cluster") \
        .getOrCreate()

# Read Spotify data
df_1 = spark.read.csv('/home/mikezhu/data/spotify_data.csv', header=True)
#.select('track_id', 'album_name', 'genres',  'artist_followers', 'artist_popularity', 'available_markets')
df_2 = spark.read.csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True).withColumnRenamed('id', 'track_id')
df_1.show(5, truncate=False)
df_2.show(5, truncate=False)

#%%
def extract_release_year(df):
    """
    Extract release year from album_release_date column
    
    Args:
        df: DataFrame with album_release_date column
        
    Returns:
        DataFrame with new release_year column
    """
    # Convert album_release_date to release_year using substring
    df_with_year = df.withColumn('release_year', 
                                F.substring('album_release_date', 1, 4))
    
    return df_with_year

df_1 = extract_release_year(df_1)



#%% Overview of dataset
df_1.printSchema()

# Extract and process genres
def extract_genres(genres_str):
    # Remove brackets and single quotes, then split by commas
    if genres_str and genres_str != '[]':
        # Remove brackets
        genres_str = genres_str.strip('[]')
        # Split by comma and clean up each genre
        return [g.strip().strip("'") for g in genres_str.split(',')]
    return []

# Count number of songs per genre
genre_counts = df_1.select('genres').rdd \
    .flatMap(lambda x: extract_genres(x.genres)) \
    .map(lambda x: (x, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: Row(genre=x[0], count=x[1]))

# Convert to DataFrame and sort
genre_counts_df = spark.createDataFrame(genre_counts) \
    .orderBy('count', ascending=False)

# Show results
print("Number of songs per genre:")
genre_counts_df.show(100, truncate=False)

# Plot top 20 genres
plt.figure(figsize=(15, 8))
top_20_genres = genre_counts_df.limit(20).toPandas()
plt.bar(top_20_genres['genre'], top_20_genres['count'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Genre')
plt.ylabel('Number of Songs')
plt.title('Top 20 Genres by Number of Songs')
plt.tight_layout()
plt.show()

df_1.count()

# %% utilities
def find_songs_by_genre(df, target_genre):
    #不只是rock 还包含 piano rock
    """
    Find all songs containing the specified genre.
    
    Args:
        df: Spark DataFrame containing songs data
        target_genre: Genre to search for
    Returns:
        Spark DataFrame containing songs of the specified genre
    """
    # Filter songs containing the target genre
    songs_with_genre = df.filter(F.col('genres').contains(target_genre))
    # Print count and percentage
    total = df.count()
    songs_with_genre.show()
    genre_count = songs_with_genre.count()
    print(f"{target_genre}: {genre_count} songs ({(genre_count/total)*100:.2f}% of total)")
    return songs_with_genre
def convert_features_to_float(df, feature_cols):
    """
    Convert selected feature columns to float type and standardize features in a distributed manner.
    
    Args:
        df: Spark DataFrame containing music features
        feature_cols: List of feature column names to convert
    Returns:
        Spark DataFrame with features converted to float type and standardized vector
    """
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    
    # fill missing values using mean
    for col in feature_cols:
        df = df.withColumn(col, F.col(col).cast('float'))
        mean_value = df.select(F.mean(col)).collect()[0][0]
        df = df.na.fill({col: mean_value})

    # Assemble features into vector column
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="unscaled_features",
        handleInvalid="skip"  # Skip invalid records
    )
    df = assembler.transform(df)

    # Cache the DataFrame to avoid recomputing 
    df = df.cache()

    # Standardize the feature vector using distributed StandardScaler
    scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="features",
        withStd=True,  # Scale to unit standard deviation
        withMean=True  # Center to zero mean
    )
    
    # Fit and transform in a distributed way
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Persist the transformed DataFrame
    df = df.persist()

    # Show schema and sample data
    print("Schema after converting to vector and standardizing:")
    df.printSchema()
    print("\nSample data after conversion:")
    df.show(5, truncate=False)
    
    return df

# %% utility PCA
def find_optimal_kmeans(df, k_values=range(3, 5)):
    """
    Find optimal k for KMeans clustering using silhouette scores
    
    Args:
        df: DataFrame with feature vectors
        k_values: Range of k values to try
        
    Returns:
        optimal_k: Optimal number of clusters
        kmeans_predictions: Predictions using optimal k
        silhouettes: List of silhouette scores for each k
    """
    silhouettes = []
    
    for k in k_values:
        # train model
        kmeans = KMeans(k=k, seed=1, featuresCol="features")
        model = kmeans.fit(df)
        
        # make predictions
        predictions = model.transform(df)
        
        # evaluate clustering
        evaluator = ClusteringEvaluator(featuresCol="features")
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
    kmeans = KMeans(k=optimal_k, seed=1, featuresCol="features")
    kmeans_model = kmeans.fit(df)
    optimal_predictions = kmeans_model.transform(df)
    
    return optimal_k, optimal_predictions, silhouettes

def analyze_pca_composition(model_pca, feature_cols):
    """
    Analyze and visualize the composition of a trained PCA model using heatmap
    
    Args:
        model_pca: Trained PCA model
        feature_cols: List of original feature names
    """
    # Get principal components matrix
    pc_matrix = model_pca.pc.toArray()
    n_components = pc_matrix.shape[1]
    
    # Create DataFrame with component compositions
    components_df = pd.DataFrame(
        pc_matrix,
        columns=[f'PC{i}' for i in range(n_components)],
        index=feature_cols
    )
    
    # Plot heatmap
    plt.figure(figsize=(6, 8))
    im = plt.imshow(components_df, cmap='RdBu', aspect='auto')
    plt.colorbar(im, label='Component Weight')
    
    # Add value annotations
    for i in range(len(feature_cols)):
        for j in range(n_components):
            plt.text(j, i, f'{components_df.iloc[i, j]:.2f}',
                    ha='center', va='center')
    
    # Customize plot
    plt.xticks(range(n_components), components_df.columns)
    plt.yticks(range(len(feature_cols)), feature_cols)
    plt.title('PCA Components Composition')
    plt.xlabel('Principal Components')
    plt.ylabel('Original Features')
    
    plt.tight_layout()
    plt.show()
    
    return components_df

def find_optimal_pca_components(df, threshold=0.95, k=None):
    """
    Find optimal number of PCA components by analyzing explained variance
    
    Args:
        df: DataFrame with feature vectors
        threshold: Variance threshold for optimal components
        k: Fixed number of components (optional)
        
    Returns:
        optimal_n: Optimal number of components
        pca_features: PCA transformed features
        explained_variances: Array of explained variance ratios
        cumulative_variance: Array of cumulative explained variance ratios
        model: Trained PCA model
    """
    n_features = df.first()["features"].size
    pca = PCA(k=n_features, inputCol="features", outputCol="pcaFeatures") 
    model = pca.fit(df)

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
    model = pca.fit(df)

    pca_results = model.transform(df)
    
    return optimal_n, pca_results, explained_variances, cumulative_variance, model

def plot_pca_sample(df_pca, sample_size=0.1, seed=None):
    """
    Plot sampled PCA points in 2D space
    
    Args:
        df_pca: DataFrame with PCA features
        sample_size: Fraction of data to sample (default 0.1)
        seed: Random seed for sampling (optional)
    """
    # Sample data
    if seed is None:
        sampled_data = df_pca.sample(False, sample_size)
    else:
        sampled_data = df_pca.sample(False, sample_size, seed=seed)
    
    # Convert to pandas and extract coordinates
    df = sampled_data.select(vector_to_array('pcaFeatures').alias('features')).toPandas()
    coords = np.vstack(df['features'].values)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'PCA Distribution of Songs (Sample Size: {sample_size*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-2.3, 1.3)
    plt.xlim(-6.3, 4.3)
    plt.tight_layout()
    plt.show()

def plot_yearly_distribution_animation(df_pca, sample_size=0.1, grid_size=100, seed=42):
    """
    Create an animated plot showing the evolution of song distribution in PCA space over time
    
    Args:
        df_pca: DataFrame with PCA features and metadata
        sample_size: Fraction of data to sample
        grid_size: Number of grid cells in each dimension
        seed: Random seed for sampling
    """
    # Sample data
    sampled_df = df_pca.sample(False, sample_size, seed=seed)
    
    # Convert to pandas
    df = sampled_df.select(
        vector_to_array('pcaFeatures').alias('features'),
        'release_year'
    ).toPandas()
    
    # Extract PCA coordinates and years
    pca_coords = np.vstack(df['features'].values)
    years = df['release_year'].values

    # Create grid for density estimation
    x_edges = np.linspace(pca_coords[:,0].min(), pca_coords[:,0].max(), grid_size)
    y_edges = np.linspace(pca_coords[:,1].min(), pca_coords[:,1].max(), grid_size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colorbar axes
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    def update(frame):
        ax.clear()
        year = sorted(np.unique(years))[frame]
        year_mask = years == year
        year_coords = pca_coords[year_mask]
        
        if len(year_coords) > 0:
            # Calculate 2D histogram
            hist, _, _ = np.histogram2d(
                year_coords[:,0], 
                year_coords[:,1],
                bins=[x_edges, y_edges]
            )
            
            # Plot heatmap
            im = ax.imshow(
                hist.T,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin='lower',
                cmap='viridis',
                aspect='auto'
            )
            
            ax.set_title(f'Year: {year}')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            
            # Update colorbar
            if frame == 0:
                cbar = fig.colorbar(im, cax=cbar_ax, label='Number of Songs')
                cbar.ax.yaxis.label.set_color('white')
                cbar.ax.tick_params(colors='white')
    
    # Create animation
    unique_years = sorted(np.unique(years))
    anim = animation.FuncAnimation(
        fig, 
        update,
        frames=len(unique_years),
        interval=100,  # 100ms between frames
        repeat=True
    )
    
    plt.suptitle('Evolution of Music Distribution in PCA Space', y=1.02, fontsize=16)

    
    # Save animation
    anim.save('music_distribution_evolution.gif', writer='pillow')
    plt.show()


def calculate_yearly_similarity(df_pca, sample_size=1.0, seed=42):
    """
    Calculate average distance to centroid for songs within each year
    
    Args:
        df_pca: DataFrame with PCA features and release year
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    Returns:
        pandas DataFrame with yearly similarity metrics
    """
    # Sample data if needed
    sampled_df = df_pca.sample(False, sample_size, seed=seed)
    
    # Group by year and calculate metrics
    yearly_metrics = sampled_df.select(
        'release_year',
        vector_to_array('pcaFeatures').alias('features')
    ).groupBy('release_year').agg(
        F.collect_list('features').alias('year_features')
    )
    
    # Calculate distances
    def compute_distances(features_list):
        features_array = np.vstack(features_list)
        centroid = np.mean(features_array, axis=0)
        distances = np.sqrt(np.sum((features_array - centroid) ** 2, axis=1))
        return float(np.mean(distances)), float(np.std(distances))
    
    compute_distances_udf = F.udf(compute_distances, ArrayType(DoubleType()))
    
    yearly_metrics = yearly_metrics.withColumn(
        'metrics',
        compute_distances_udf('year_features')
    ).select(
        'release_year',
        F.col('metrics')[0].alias('avg_distance'),
        F.col('metrics')[1].alias('std_distance')
    )
    
    # Convert to pandas for plotting
    yearly_df = yearly_metrics.toPandas()
    yearly_df['release_year'] = pd.to_numeric(yearly_df['release_year'])
    yearly_df = yearly_df.sort_values('release_year')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        yearly_df['release_year'], 
        yearly_df['avg_distance'],
        yerr=yearly_df['std_distance'],
        capsize=3,
        alpha=0.7
    )
    plt.fill_between(
        yearly_df['release_year'],
        yearly_df['avg_distance'] - yearly_df['std_distance'],
        yearly_df['avg_distance'] + yearly_df['std_distance'],
        alpha=0.2
    )
    
    # Fit trend line
    plt.plot(yearly_df['release_year'], p(yearly_df['release_year']), "r--", alpha=0.8)
    
    plt.xlabel('Year')
    plt.ylabel('Average Distance to Centroid')
    plt.title('Musical Similarity Over Time\n(Lower Distance = More Similar Songs)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return yearly_df

#%% run PCA+sample+KMeans

# Example usage
selected_songs = find_songs_by_genre(df_1, 'rock')
feature_cols = [
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

# Convert features to float and standardize
df = convert_features_to_float(selected_songs,feature_cols)

# 1. PCA: find optimal number of components
optimal_n, df_pca, explained_variances, cumulative_variance, model_pca = find_optimal_pca_components(df)
df_pca.persist()

# PCA composition
components_df = analyze_pca_composition(model_pca, feature_cols)

# PCA sample
plot_pca_sample(df_pca, sample_size=0.01)

# PCA yearly distribution animation
plot_yearly_distribution_animation(df_pca, sample_size=0.1)

# Usage example:
df_viz=df_pca.filter(F.col('release_year') >= 1940)
yearly_similarity = calculate_yearly_similarity(df_viz, sample_size=0.1)


# %%
