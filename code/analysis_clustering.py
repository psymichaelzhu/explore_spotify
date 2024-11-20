
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
import matplotlib.animation as animation


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
def find_optimal_kmeans(features, k_values=range(3, 5)):
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

#%% merge cluster results
# merge to get cluster results
merged_results = kmeans_predictions_pca.withColumn("tmp_id", F.monotonically_increasing_id()) \
            .join(df_features.withColumn("tmp_id", F.monotonically_increasing_id()).withColumnRenamed("features", "raw_features"), on="tmp_id", how="inner").drop("tmp_id") \
            .join(df,on=["id","name","artist"],how="inner")
merged_results.show()
merged_results.count()
#examine cluster distribution
merged_results.groupby('prediction') \
               .count() \
               .show()
cluster_results=merged_results.filter(F.col("prediction") != 3)
cluster_results.groupby('prediction') \
               .count() \
               .show()


#%% visualization functions
def exact_to_pd(cluster_full_data,artist_name=None,sample_size=0.1,seed=None):
    '''extract a subset of data, for visualization'''
    num_clusters = cluster_full_data.select('prediction').distinct().count()
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    if artist_name is None:
        if seed is None:
            cluster_data = cluster_full_data.sample(False, sample_size)  
        else:
            cluster_data = cluster_full_data.sample(False, sample_size, seed=seed)  
    else:
        cluster_data = cluster_full_data.filter(F.col("artist") == artist_name)
    cluster_data = cluster_data.toPandas()
    cluster_data['year'] = pd.to_datetime(cluster_data['release_date']).dt.year
    return num_clusters,colors,cluster_data

def plot_cluster_distribution(cluster_data,artist_name=None,sample_size=0.1,seed=None):
    """Plot distribution of songs across clusters"""
    num_clusters,colors,cluster_data=exact_to_pd(cluster_data,artist_name,sample_size,seed)
 
    all_clusters = pd.Series(0, index=range(num_clusters))
    cluster_counts = cluster_data['prediction'].value_counts()
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

def plot_cluster_evolution(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """Plot evolution of clusters over time"""
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Calculate proportion of each cluster per year
    yearly_proportions = cluster_data.pivot_table(
        index='year',
        columns='prediction',
        aggfunc='size',
        fill_value=0
    )
    yearly_proportions = yearly_proportions.div(yearly_proportions.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each cluster separately to ensure correct color mapping
    bottom = np.zeros(len(yearly_proportions))
    for cluster in range(num_clusters):
        if cluster in yearly_proportions.columns:
            values = yearly_proportions[cluster].values
            ax.fill_between(yearly_proportions.index, bottom, bottom + values, 
                          alpha=0.7, label=f'Cluster {cluster}',
                          color=colors[cluster])
            bottom += values
    
    plt.xlabel('Year')
    plt.ylabel('Proportion of Songs')
    plt.title('Evolution of Song Clusters Over Time')
    plt.legend()
    plt.show()

def plot_cluster_scatter(cluster_data, artist_name=None, dim_1=0, dim_2=1,sample_size=0.1,seed=None):
    """Plot scatter of clusters in PCA space"""
    num_clusters, colors,cluster_data=exact_to_pd(cluster_data, artist_name,sample_size,seed)
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h']
    plt.figure(figsize=(10, 8))
    for cluster in range(num_clusters):
        plot_data = cluster_data[cluster_data["prediction"] == cluster]
        plt.scatter(plot_data["features"].apply(lambda x: float(x[dim_1])),
                   plot_data["features"].apply(lambda x: float(x[dim_2])),
                   c=[colors[cluster]], 
                   marker=markers[cluster % len(markers)],
                   label=f'Cluster {cluster}',
                   s=100)
        for _, row in plot_data.iterrows():
            plt.annotate(row["name"], 
                        (float(row["features"][dim_1]), float(row["features"][dim_2])),
                    xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                        fontsize=8)
    plt.xlabel(f"Component {dim_1}")
    plt.ylabel(f"Component {dim_2}")
    plt.title("Songs Clustered in Dimensionally Reduced Space")
    plt.legend()
    plt.show()

def plot_cluster_radar(cluster_data, artist_name=None,sample_size=0.1,seed=None):
    """
    Create a radar plot to compare cluster centroids with standardized features
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
    """
    # Extract data and get number of clusters
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name,sample_size,seed)
    
    # Get feature values from the features column
    feature_values = pd.DataFrame(cluster_data['raw_features'].tolist(), 
                                columns=feature_cols)
    
    # Calculate global means and standard deviations for standardization
    global_means = feature_values[feature_cols].mean()
    global_stds = feature_values[feature_cols].std()
    
    # Standardize features
    standardized_values = feature_values.copy()
    for col in feature_cols:
        if global_stds[col] > 0:  # Avoid division by zero
            standardized_values[col] = (feature_values[col] - global_means[col]) / global_stds[col]
        else:
            standardized_values[col] = N
    
    # Add prediction column
    standardized_values['prediction'] = cluster_data['prediction']
    
    # Calculate mean values for each cluster using standardized features
    cluster_means = standardized_values.groupby('prediction')[feature_cols].mean()
    
    # Set up the angles for radar plot
    angles = np.linspace(0, 2*np.pi, len(feature_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the circle
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data for each cluster
    for cluster in range(num_clusters):
        if cluster in cluster_means.index:
            values = cluster_means.loc[cluster].values
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f'Cluster {cluster}', 
                   color=colors[cluster])
            ax.fill(angles, values, alpha=0.25, color=colors[cluster])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    title = "Cluster Characteristics"
    if artist_name:
        title += f" for {artist_name}"
    plt.title(title, fontsize=17)
    plt.tight_layout()
    plt.show()
    
    return cluster_means


#%% dynamic evolution scatter plot
import matplotlib.animation as animation

def plot_cluster_animation(cluster_data, artist_name=None, dim_1=0, dim_2=1, sample_size=0.1, seed=None, interval=1000):
    """
    Create an animated scatter plot showing cluster evolution over time
    """
    # Get data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Sort by year for animation
    cluster_data = cluster_data.sort_values('year')
    years = sorted(cluster_data['year'].unique())
    
    # Set up the figure and animation
    fig, ax = plt.subplots(figsize=(12, 8))
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h']
    
    def animate(frame_year):
        ax.clear()
        
        # Plot historical data (previous years) with lower alpha
        historical_data = cluster_data[cluster_data['year'] < frame_year]
        if not historical_data.empty:
            for cluster in range(num_clusters):
                plot_data = historical_data[historical_data["prediction"] == cluster]
                if not plot_data.empty:
                    ax.scatter(plot_data["features"].apply(lambda x: float(x[dim_1])),
                             plot_data["features"].apply(lambda x: float(x[dim_2])),
                             c=[colors[cluster]], 
                             marker=markers[cluster % len(markers)],
                             label=f'Cluster {cluster} (historical)',
                             alpha=0.3,  # Lower alpha for historical data
                             s=80)
        
        # Plot current year data with higher alpha and larger markers
        current_data = cluster_data[cluster_data['year'] == frame_year]
        if not current_data.empty:
            for cluster in range(num_clusters):
                plot_data = current_data[current_data["prediction"] == cluster]
                if not plot_data.empty:
                    ax.scatter(plot_data["features"].apply(lambda x: float(x[dim_1])),
                             plot_data["features"].apply(lambda x: float(x[dim_2])),
                             c=[colors[cluster]], 
                             marker=markers[cluster % len(markers)],
                             label=f'Cluster {cluster} ({frame_year})',
                             alpha=1.0,  # Full alpha for current year
                             s=150,  # Larger markers for current year
                             edgecolor='black')
                    
                    # Add song names for current year
                    for _, row in plot_data.iterrows():
                        ax.annotate(f"{row['name']} ({frame_year})",
                                  (float(row["features"][dim_1]), float(row["features"][dim_2])),
                                  xytext=(5, 5),
                                  textcoords='offset points',
                                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                                  fontsize=8)
        
        # Set labels and title
        ax.set_xlabel(f"Component {dim_1}")
        ax.set_ylabel(f"Component {dim_2}")
        title = f"Songs Clustered in PCA Space\nYear: {frame_year}"
        if artist_name:
            title = f"{artist_name}: {title}"
        ax.set_title(title)
        
        # Add year count information
        current_count = len(current_data)
        total_count = len(cluster_data[cluster_data['year'] <= frame_year])
        ax.text(0.02, 0.98, f"New songs in {frame_year}: {current_count}\nTotal songs: {total_count}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='top')
        
        # Set consistent axis limits
        all_x = cluster_data["features"].apply(lambda x: float(x[dim_1]))
        all_y = cluster_data["features"].apply(lambda x: float(x[dim_2]))
        ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
        ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)
        
        # Customize legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    if not years:
        print("No data available for animation")
        return None
    
    ani = animation.FuncAnimation(fig, animate, frames=years,
                                interval=interval, repeat=True)
    
    # Save animation
    try:
        ani.save(f'cluster_animation{"_" + artist_name if artist_name else ""}.gif',
                writer='pillow', fps=1)
        print(f"Animation saved successfully!")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.show()
    return ani

# Example usage:
try:
    ani = plot_cluster_animation(cluster_results, "Metallica", interval=1000)
    if ani is None:
        print("Animation could not be created")
except Exception as e:
    print(f"Error creating animation: {e}")

#%% visualization

#global
plot_cluster_distribution(cluster_results)
plot_cluster_evolution(cluster_results)
plot_cluster_radar(cluster_results)
#plot_cluster_scatter(cluster_results) # too many points
#individual
plot_cluster_distribution(cluster_results,"Coldplay")
plot_cluster_evolution(cluster_results,"Coldplay")
plot_cluster_scatter(cluster_results,"Coldplay")
plot_cluster_radar(cluster_results,"Coldplay")

plot_cluster_distribution(cluster_results,"Taylor Swift")
plot_cluster_evolution(cluster_results,"Taylor Swift")
plot_cluster_scatter(cluster_results,"Taylor Swift")
plot_cluster_radar(cluster_results,"Taylor Swift")




# %% more artists
plot_cluster_scatter(cluster_results,"Oasis")
plot_cluster_radar(cluster_results,"Oasis")
plot_cluster_evolution(cluster_results,"Oasis")
plot_cluster_distribution(cluster_results,"Oasis")
plot_cluster_distribution(cluster_results,"Metallica")
plot_cluster_scatter(cluster_results,"Metallica")
plot_cluster_radar(cluster_results,"Metallica")
plot_cluster_evolution(cluster_results,"Metallica")
plot_cluster_distribution(cluster_results,"Pink Floyd")
plot_cluster_scatter(cluster_results,"Pink Floyd")
plot_cluster_radar(cluster_results,"Pink Floyd")
plot_cluster_evolution(cluster_results,"Pink Floyd")
plot_cluster_distribution(cluster_results,"Radiohead")
plot_cluster_scatter(cluster_results,"Radiohead")
plot_cluster_radar(cluster_results,"Radiohead")
plot_cluster_evolution(cluster_results,"Radiohead")
plot_cluster_distribution(cluster_results,"Blur")
plot_cluster_scatter(cluster_results,"Blur")
plot_cluster_radar(cluster_results,"Blur")
plot_cluster_evolution(cluster_results,"Blur")

# %% similarity analysis based on centroid evolution of PCA space

def analyze_artist_evolution(cluster_results, artist_name, time_decay=0.9):
    """
    Analyze artist evolution using cluster centroids and time-weighted similarities
    
    Args:
        cluster_results: DataFrame with PCA features and metadata
        artist_name: Name of the artist to analyze
        time_decay: Decay factor for historical similarities (0-1)
    """
    # Extract artist data and convert to pandas
    artist_data = cluster_results.filter(F.col("artist") == artist_name).toPandas()
    artist_data['year'] = pd.to_datetime(artist_data['release_date']).dt.year
    artist_data['feature_array'] = artist_data['features'].apply(lambda x: np.array(x.toArray()))
    
    # Initialize results
    yearly_cohesion = {}
    evolution_scores = {}
    yearly_centroids = {}
    
    # Calculate yearly centroids and cohesion
    for year in sorted(artist_data['year'].unique()):
        year_songs = artist_data[artist_data['year'] == year]
        features = np.stack(year_songs['feature_array'].values)
        n_songs = len(year_songs)
        
        # Calculate centroid
        centroid = np.mean(features, axis=0)
        yearly_centroids[year] = centroid
        
        # Calculate cohesion (average distance to centroid)
        if n_songs == 1:
            mean_distance = np.nan  # Set to 1 if only one song
        else:
            distances = [np.linalg.norm(f - centroid) for f in features]
            mean_distance = np.mean(distances)
            
        yearly_cohesion[year] = {
            'mean_distance': mean_distance,
            'n_songs': n_songs
        }
    
    # Calculate evolution scores
    years = sorted(yearly_centroids.keys())
    for i, current_year in enumerate(years[1:], 1):
        current_centroid = yearly_centroids[current_year]
        
        weighted_similarities = []
        weights = []
        
        for past_year in years[:i]:
            past_centroid = yearly_centroids[past_year]
            time_diff = current_year - past_year
            weight = time_decay ** time_diff
            
            similarity = np.dot(current_centroid, past_centroid) / \
                        (np.linalg.norm(current_centroid) * np.linalg.norm(past_centroid))
            
            weighted_similarities.append(similarity * weight)
            weights.append(weight)
        
        evolution_scores[current_year] = {
            'evolution_score': 1 - (sum(weighted_similarities) / sum(weights)),
            'n_previous_years': len(weighted_similarities)
        }
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cluster Cohesion
    years = list(yearly_cohesion.keys())
    means = [d['mean_distance'] for d in yearly_cohesion.values()]
    
    ax1.plot(years, means, 'o-')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Distance to Centroid')
    ax1.set_title(f'{artist_name}: Yearly Cluster Cohesion')
    
    # Plot 2: Evolution Scores
    years = list(evolution_scores.keys())
    scores = [d['evolution_score'] for d in evolution_scores.values()]
    
    ax2.plot(years, scores, 'o-')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Evolution Score')
    ax2.set_title(f'{artist_name}: Musical Evolution Score')
    
    plt.tight_layout()
    plt.show()
    
    return yearly_cohesion, evolution_scores

# Example usage:
cohesion, evolution = analyze_artist_evolution(cluster_results, "Coldplay")

# %%

def analyze_artist_evolution(cluster_results, artist_name, time_decay=0.9):
    """
    Analyze artist evolution using cluster centroids and time-weighted similarities
    
    Args:
        cluster_results: DataFrame with PCA features and metadata
        artist_name: Name of the artist to analyze
        time_decay: Decay factor for historical similarities (0-1)
    """
    # Extract artist data and convert to pandas
    artist_data = cluster_results.filter(F.col("artist") == artist_name).toPandas()
    artist_data['year'] = pd.to_datetime(artist_data['release_date']).dt.year
    artist_data['feature_array'] = artist_data['features'].apply(lambda x: np.array(x.toArray()))
    
    # Initialize results
    yearly_cohesion = {}
    evolution_scores = {}
    yearly_centroids = {}
    
    # Calculate yearly centroids and cohesion
    for year in sorted(artist_data['year'].unique()):
        year_songs = artist_data[artist_data['year'] == year]
        features = np.stack(year_songs['feature_array'].values)
        n_songs = len(year_songs)
        
        # Calculate centroid
        centroid = np.mean(features, axis=0)
        yearly_centroids[year] = centroid
        
        # Calculate cohesion (average distance to centroid)
        if n_songs == 1:
            mean_distance = np.nan  # Set to 1 if only one song
        else:
            distances = [np.linalg.norm(f - centroid) for f in features]
            mean_distance = np.mean(distances)
            
        yearly_cohesion[year] = {
            'mean_distance': mean_distance,
            'n_songs': n_songs
        }
    
    # Calculate evolution scores
    years = sorted(yearly_centroids.keys())
    for i, current_year in enumerate(years[1:], 1):
        current_centroid = yearly_centroids[current_year]
        
        weighted_similarities = []
        weights = []
        
        for past_year in years[:i]:
            past_centroid = yearly_centroids[past_year]
            time_diff = current_year - past_year
            weight = time_decay ** time_diff
            
            similarity = np.dot(current_centroid, past_centroid) / \
                        (np.linalg.norm(current_centroid) * np.linalg.norm(past_centroid))
            
            weighted_similarities.append(similarity * weight)
            weights.append(weight)
        
        evolution_scores[current_year] = {
            'evolution_score': 1 - (sum(weighted_similarities) / sum(weights)),
            'n_previous_years': len(weighted_similarities)
        }
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cluster Cohesion
    years = list(yearly_cohesion.keys())
    means = [d['mean_distance'] for d in yearly_cohesion.values()]
    
    ax1.plot(years, means, 'o-')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Distance to Centroid')
    ax1.set_title(f'{artist_name}: Yearly Cluster Cohesion')
    
    # Plot 2: Evolution Scores
    years = list(evolution_scores.keys())
    scores = [d['evolution_score'] for d in evolution_scores.values()]
    
    ax2.plot(years, scores, 'o-')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Evolution Score')
    ax2.set_title(f'{artist_name}: Musical Evolution Score')
    
    plt.tight_layout()
    plt.show()
    
    return yearly_cohesion, evolution_scores

# Example usage:
cohesion, evolution = analyze_artist_evolution(cluster_results, "Coldplay")