
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
    #'duration_ms',
    'loudness',
    'key',
    'tempo', 
    'time_signature',
    'mode',
    'acousticness', 
    'instrumentalness',  
    'speechiness',
    #'danceability', 
    #'energy',
    #'valence'
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

#%% visualization

#global
plot_cluster_distribution(cluster_results)
plot_cluster_evolution(cluster_results)
plot_cluster_radar(cluster_results)
'''
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

'''


# %% more artists
'''
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
'''


#%% animation based on centroid evolution of PCA space
import matplotlib.animation as animation

def plot_cluster_animation(cluster_data, artist_name=None, dim_1=0, dim_2=1, sample_size=0.1, seed=None, interval=1000):
    """
    Create an animated scatter plot showing centroid evolution over time, with innovation metrics
    """
    # Get data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Sort by year for animation
    cluster_data = cluster_data.sort_values('year')
    years = sorted(cluster_data['year'].unique())
    
    # First create static trend plot
    fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
    within_year_distances = []
    between_year_distances = []
    
    historical_data = pd.DataFrame()
    
    for year in years:
        current_data = cluster_data[cluster_data['year'] == year]
        if not current_data.empty:
            current_x = current_data["features"].apply(lambda x: float(x[dim_1]))
            current_y = current_data["features"].apply(lambda x: float(x[dim_2]))
            current_centroid_x = current_x.mean()
            current_centroid_y = current_y.mean()
            
            # Calculate within-year innovation: average distance from songs to current centroid
            within_year_dist = np.mean([np.sqrt((x - current_centroid_x)**2 + (y - current_centroid_y)**2) 
                                      for x, y in zip(current_x, current_y)])
            within_year_distances.append(within_year_dist)
            
            # Calculate between-year innovation: distance from current to historical centroid
            if not historical_data.empty:
                historical_centroid_x = historical_data["features"].apply(lambda x: float(x[dim_1])).mean()
                historical_centroid_y = historical_data["features"].apply(lambda x: float(x[dim_2])).mean()
                between_year_dist = np.sqrt((current_centroid_x - historical_centroid_x)**2 + 
                                          (current_centroid_y - historical_centroid_y)**2)
            else:
                between_year_dist = np.nan
            between_year_distances.append(between_year_dist)
            
            historical_data = pd.concat([historical_data, current_data])
    
    # Plot static trend lines
    ax_trend.plot(years, within_year_distances, 'b-', label='Within-year Innovation', linewidth=2)
    ax_trend.plot(years, between_year_distances, 'r-', label='Between-year Innovation', linewidth=2)
    ax_trend.set_xlabel('Year')
    ax_trend.set_ylabel('Innovation Distance')
    ax_trend.set_ylim(0, 2.5)
    ax_trend.axhline(y=1, color='black', linestyle='--', linewidth=1)
    title = "Innovation Trends Over Time"
    if artist_name:
        title = f"{artist_name}: {title}"
    ax_trend.set_title(title)
    ax_trend.legend()
    plt.tight_layout()
    plt.show()
    
    # Set up the animation figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])  # Main scatter plot
    ax2 = fig.add_subplot(gs[1])  # Innovation metrics plot
    
    # Store innovation metrics for animation
    within_year_distances = []
    between_year_distances = []
    years_for_plot = []
    
    def animate(frame_year):
        ax1.clear()
        ax2.clear()
        
        # Plot historical data and centroid
        historical_data = cluster_data[cluster_data['year'] < frame_year]
        if not historical_data.empty:
            # Plot historical points
            ax1.scatter(historical_data["features"].apply(lambda x: float(x[dim_1])),
                      historical_data["features"].apply(lambda x: float(x[dim_2])),
                      c='gray',
                      alpha=0.3,
                      s=80)
            
            # Calculate and plot historical centroid
            historical_centroid_x = historical_data["features"].apply(lambda x: float(x[dim_1])).mean()
            historical_centroid_y = historical_data["features"].apply(lambda x: float(x[dim_2])).mean()
            ax1.scatter(historical_centroid_x, historical_centroid_y,
                      c='gray',
                      marker='*',
                      s=200,
                      alpha=0.5,
                      edgecolor='black',
                      linewidth=1,
                      label='Historical Centroid')
        
        # Plot current year data and centroid
        current_data = cluster_data[cluster_data['year'] == frame_year]
        if not current_data.empty:
            # Plot current year points
            current_x = current_data["features"].apply(lambda x: float(x[dim_1]))
            current_y = current_data["features"].apply(lambda x: float(x[dim_2]))
            ax1.scatter(current_x, current_y,
                      c='blue',
                      alpha=1.0,
                      s=150,
                      edgecolor='black',
                      label=f'Songs ({frame_year})')
            
            # Calculate and plot current year centroid
            current_centroid_x = current_x.mean()
            current_centroid_y = current_y.mean()
            ax1.scatter(current_centroid_x, current_centroid_y,
                      c='red',
                      marker='*',
                      s=300,
                      alpha=1.0,
                      edgecolor='black',
                      linewidth=2,
                      label=f'Current Centroid ({frame_year})')
            
            # Calculate innovation metrics
            # Within-year innovation: average distance from songs to current centroid
            within_year_dist = np.mean([np.sqrt((x - current_centroid_x)**2 + (y - current_centroid_y)**2) 
                                          for x, y in zip(current_x, current_y)])

            # Between-year innovation: distance from current to historical centroid
            if not historical_data.empty:
                between_year_dist = np.sqrt((current_centroid_x - historical_centroid_x)**2 + 
                                          (current_centroid_y - historical_centroid_y)**2)
                between_year_distances.append(between_year_dist)
            else:
                between_year_dist = np.nan
                between_year_distances.append(between_year_dist)
                
            within_year_distances.append(within_year_dist)
            years_for_plot.append(frame_year)
            
            # Plot innovation metrics
            ax2.plot(years_for_plot, within_year_distances, 'b-', label='Within-year Innovation')
            ax2.plot(years_for_plot, between_year_distances, 'r-', label='Between-year Innovation')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Innovation Distance')
            ax2.set_title('Innovation Metrics Over Time')
            ax2.legend()
            ax2.grid(True)
            ax2.set_ylim(0, 2.5)
            ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
            
            # Add song names for current year
            for _, row in current_data.iterrows():
                ax1.annotate(f"{row['name']} ({frame_year})",
                          (float(row["features"][dim_1]), float(row["features"][dim_2])),
                          xytext=(5, 5),
                          textcoords='offset points',
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                          fontsize=8)
        
        # Set labels and title for main plot
        ax1.set_xlabel(f"Component {dim_1}")
        ax1.set_ylabel(f"Component {dim_2}")
        title = f"Songs and Centroids in PCA Space\nYear: {frame_year}"
        if artist_name:
            title = f"{artist_name}: {title}"
        ax1.set_title(title)
        
        # Add year count information
        current_count = len(current_data)
        total_count = len(cluster_data[cluster_data['year'] <= frame_year])
        ax1.text(0.02, 0.98, f"New songs in {frame_year}: {current_count}\nTotal songs: {total_count}",
                transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='top')
        
        # Set consistent axis limits
        all_x = cluster_data["features"].apply(lambda x: float(x[dim_1]))
        all_y = cluster_data["features"].apply(lambda x: float(x[dim_2]))
        ax1.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
        ax1.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)
        
        # Customize legend
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    if not years:
        print("No data available for animation")
        return None
    
    ani = animation.FuncAnimation(fig, animate, frames=years,
                                interval=interval, repeat=True)
    
    # Save animation
    try:
        ani.save(f'centroid_evolution{"_" + artist_name if artist_name else ""}.gif',
                writer='pillow', fps=1)
        print(f"Animation saved successfully!")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.show()
    return ani

# Example usage:
'''
try:
    ani = plot_cluster_animation(cluster_results, "Metallica", interval=1000)
    if ani is None:
        print("Animation could not be created")
except Exception as e:
    print(f"Error creating animation: {e}")
'''
# %% distance distribution
def plot_distance_distribution(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """
    Plot yearly distribution of songs' distances to centroids
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Process data year by year
    years = sorted(cluster_data['year'].unique())
    historical_centroid = None
    
    # Store distances for boxplot
    current_distances_by_year = {year: [] for year in years}
    historical_distances_by_year = {year: [] for year in years}
    
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            # Extract feature coordinates
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            
            # Calculate current year centroid
            current_centroid = current_points.mean(axis=0)
            
            # Calculate distances to current centroid
            current_distances = np.sqrt(
                np.sum((current_points - current_centroid) ** 2, axis=1)
            current_distances_by_year[year] = current_distances
            
            # Calculate distances to historical centroid if available
            if historical_centroid is not None:
                historical_distances = np.sqrt(
                    np.sum((current_points - historical_centroid) ** 2, axis=1)
                historical_distances_by_year[year] = historical_distances
            
            # Update historical centroid (cumulative mean)
            if historical_centroid is None:
                historical_centroid = current_centroid
            else:
                historical_data = cluster_data[cluster_data['year'] <= year]
                historical_points = np.array([
                    [float(features[0]), float(features[1])] 
                    for features in historical_data['features']
                ])
                historical_centroid = historical_points.mean(axis=0)
    
    # Create box plots
    box_data_current = [current_distances_by_year[year] for year in years]
    box_data_historical = [historical_distances_by_year[year] for year in years]
    
    # Plot current year distances
    # Plot points
    for i, year_data in enumerate(box_data_current):
        if len(year_data) > 0:
            x = np.full_like(year_data, years[i])
            ax1.scatter(x, year_data, alpha=0.3, s=50,color="#1E337D")
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Distance to Current Year Centroid')
    ax1.set_ylim(0, 5)
    title1 = "Distribution of Distances to Current Year Centroid"
    if artist_name:
        title1 = f"{artist_name}: {title1}"
    ax1.set_title(title1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot historical distances
    # Plot points
    for i, year_data in enumerate(box_data_historical):
        if len(year_data) > 0:
            x = np.full_like(year_data, years[i])
            ax2.scatter(x, year_data, alpha=0.3, s=50,color="#BE512E")
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Distance to Historical Centroid')
    ax2.set_ylim(0, 5)
    title2 = "Distribution of Distances to Historical Centroid"
    if artist_name:
        title2 = f"{artist_name}: {title2}"
    ax2.set_title(title2)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

#plot_distance_distribution(cluster_results,sample_size=0.01)

'''
plot_distance_distribution(cluster_results, "Metallica")
plot_distance_distribution(cluster_results, "Radiohead")
plot_distance_distribution(cluster_results, "Taylor Swift")
plot_distance_distribution(cluster_results, "Oasis")
plot_distance_distribution(cluster_results, "Coldplay")
plot_distance_distribution(cluster_results, "Avril Lavigne")
plot_distance_distribution(cluster_results, "Lady Gaga")
plot_distance_distribution(cluster_results, "Ariana Grande")
plot_distance_distribution(cluster_results, "Ed Sheeran")
plot_distance_distribution(cluster_results, "Justin Bieber")
plot_distance_distribution(cluster_results, "Drake")
plot_distance_distribution(cluster_results, "Billie Eilish")
plot_distance_distribution(cluster_results, "Eminem")
plot_distance_distribution(cluster_results, "Bruno Mars")
plot_distance_distribution(cluster_results, "Linkin Park")
plot_distance_distribution(cluster_results, "OneRepublic")
plot_distance_distribution(cluster_results, "One Direction")
'''
# %% innovation levels over time
def plot_innovation_levels_over_time(cluster_data, artist_name=None, sample_size=0.1, seed=None, 
                                   bins=[0, 1, 2, 3, 4], distance_type='historical'):
    """
    Plot the number of songs in different innovation level ranges over time and return the data
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
        bins: Innovation level ranges (default: [0, 1, 2, 3, 4])
        distance_type: Type of distance to calculate ('historical' or 'internal')
        
    Returns:
        DataFrame containing innovation counts by year and range
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Process data year by year
    years = sorted(cluster_data['year'].unique())
    historical_centroid = None
    
    # Store innovation levels for each year
    innovation_by_year = {year: [] for year in years}
    
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            
            if distance_type == 'historical':
                if historical_centroid is not None:
                    # Calculate distances to historical centroid
                    distances = np.sqrt(
                        np.sum((current_points - historical_centroid) ** 2, axis=1)
                    innovation_by_year[year].extend(distances)
                
                # Update historical centroid
                if historical_centroid is None:
                    historical_centroid = current_centroid
                else:
                    historical_data = cluster_data[cluster_data['year'] <= year]
                    historical_points = np.array([
                        [float(features[0]), float(features[1])] 
                        for features in historical_data['features']
                    ])
                    historical_centroid = historical_points.mean(axis=0)
            
            else:  # internal distance
                # Calculate distances to current year's centroid
                distances = np.sqrt(
                    np.sum((current_points - current_centroid) ** 2, axis=1)
                innovation_by_year[year].extend(distances)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Calculate counts for each bin range
    bin_ranges = list(zip(bins[:-1], bins[1:]))
    counts_by_range = {f"{low:.1f}-{high:.1f}": [] for low, high in bin_ranges}
    
    for year in years:
        year_innovations = innovation_by_year[year]
        if year_innovations:
            # Count songs in each range
            for (low, high) in bin_ranges:
                count = sum((low <= x < high) for x in year_innovations)
                counts_by_range[f"{low:.1f}-{high:.1f}"].append(count)
        else:
            # Add 0 if no songs in this year
            for range_key in counts_by_range:
                counts_by_range[range_key].append(0)
    
    # Plot lines for each range
    colors = plt.cm.viridis(np.linspace(0, 1, len(bin_ranges)))
    for (range_key, counts), color in zip(counts_by_range.items(), colors):
        plt.plot(years, counts, '-o', label=f'Innovation {range_key}', 
                color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Year')
    plt.ylabel('Number of Songs')
    title = f"Distribution of {'Historical' if distance_type == 'historical' else 'Internal'} Innovation Levels Over Time"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Create and return DataFrame with results
    results_df = pd.DataFrame(counts_by_range, index=years)
    results_df.index.name = 'Year'
    
    # Add raw innovation values
    results_df['raw_innovation_values'] = [innovation_by_year[year] for year in years]
    
    return results_df

# 使用历史距离
results_df = plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.1,bins=[0,1,1.5,2,2.5,100])
plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.1,bins=[0,2,100])

# 使用年内距离
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.3,bins=[0,1,1.5,2,2.5,100])
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.3,bins=[0,2,100])


# %% proportion of highly innovative songs over time

def plot_innovation_ratio_over_time(cluster_data, threshold=2.0, artist_name=None, sample_size=0.1, seed=None, 
                                  distance_type='historical', window_size=5, year_range=(1920, 2020)):
    """
    Plot the ratio of highly innovative songs over time
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        threshold: Distance threshold for considering a song highly innovative
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
        distance_type: Type of distance to calculate ('historical' or 'internal')
        window_size: Size of moving average window
        year_range: Tuple of (start_year, end_year)
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Filter data by year range
    cluster_data = cluster_data[
        (cluster_data['year'] >= year_range[0]) & 
        (cluster_data['year'] <= year_range[1])
    ]
    
    # Process data year by year
    years = sorted(cluster_data['year'].unique())
    historical_centroid = None
    
    # Store innovation ratios and song counts for each year
    innovation_ratios = []
    song_counts = []
    
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            
            if distance_type == 'historical':
                if historical_centroid is not None:
                    distances = np.sqrt(np.sum((current_points - historical_centroid) ** 2, axis=1))
                    
                    # Calculate ratio of highly innovative songs
                    innovative_ratio = np.mean(distances >= threshold)
                    innovation_ratios.append(innovative_ratio)
                else:
                    innovation_ratios.append(0)
                
                # Update historical centroid
                if historical_centroid is None:
                    historical_centroid = current_centroid
                else:
                    historical_data = cluster_data[cluster_data['year'] <= year]
                    historical_points = np.array([
                        [float(features[0]), float(features[1])] 
                        for features in historical_data['features']
                    ])
                    historical_centroid = historical_points.mean(axis=0)
            else:  # internal distance
                distances = np.sqrt(np.sum((current_points - current_centroid) ** 2, axis=1))
                innovative_ratio = np.mean(distances >= threshold)
                innovation_ratios.append(innovative_ratio)
    
    # Calculate moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    ma_ratios = moving_average(innovation_ratios, window_size)
    ma_years = years[window_size-1:]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot original data as scatter points
    plt.scatter(years, innovation_ratios, alpha=0.3, label='Yearly ratio', color='gray')
    
    # Plot moving average line
    plt.plot(ma_years, ma_ratios, '-', label=f'{window_size}-year moving average', 
            color='blue', linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Ratio of Highly Innovative Songs')
    title = f"Ratio of {'Historical' if distance_type == 'historical' else 'Internal'} Innovation Over Time"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='x', rotation=45)
    plt.ylim(0, 1)
    
    # Add threshold value to plot
    plt.text(0.02, 0.98, f'Innovation threshold: {threshold}', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return years, innovation_ratios

# Example usage:
# Historical distance
'''
plot_innovation_ratio_over_time(cluster_results, threshold=2.0, 
                              distance_type='historical', sample_size=0.1)

# Internal distance
plot_innovation_ratio_over_time(cluster_results, threshold=2.0, 
                              distance_type='internal', sample_size=0.3)
'''



#%%
#绘制results_df
# Plot results by innovation level over years
plt.figure(figsize=(12, 6))

# Get innovation level columns (excluding raw_innovation_values)
level_columns = [col for col in results_df.columns if col != 'raw_innovation_values']

# Plot stacked bars for each level
bottom = np.zeros(len(results_df))
colors = plt.cm.viridis(np.linspace(0, 1, len(level_columns)))

for col, color in zip(level_columns, colors):
    plt.bar(results_df.index, results_df[col], bottom=bottom, 
            label=f'Innovation {col}', color=color)
    bottom += results_df[col]

plt.xlabel('Year')
plt.ylabel('Number of Songs')
plt.title('Distribution of Innovation Levels by Year')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# %%

from prophet import Prophet

# 格式化数据为 Prophet 格式
df_prophet = results_df.reset_index()
df_prophet = df_prophet.rename(columns={"Year": "ds", "0.0–1.0": "y"})

# 定义 Prophet 模型
model = Prophet()
model.fit(df_prophet)

# 预测未来 10 年
future = model.make_future_dataframe(periods=10, freq='Y')
forecast = model.predict(future)

# 可视化预测结果
model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()


# %%
def plot_centroid_evolution(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """
    Plot the evolution of centroids in PCA space over years
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Process data year by year
    years = sorted(cluster_data['year'].unique())
    centroids = []
    
    # Calculate centroid for each year
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            centroids.append((year, current_centroid))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create color gradient based on years
    num_years = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, num_years))
    
    # Plot centroids with color gradient
    for i, (year, centroid) in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], c=[colors[i]], s=100, alpha=0.8)
        plt.annotate(str(year), (centroid[0], centroid[1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add colorbar to show year progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min(years), vmax=max(years)))
    plt.colorbar(sm, label='Year')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    title = "Evolution of Music Style Centroids Over Years"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_centroid_evolution(cluster_results, sample_size=0.1)

# %%
def plot_centroid_evolution(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """
    Plot the evolution of music style centroids over years with animation effect
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    """
    import matplotlib.animation as animation
    
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Process data year by year
    years = sorted(cluster_data['year'].unique())
    centroids = []
    
    # Calculate centroid for each year
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            centroids.append((year, current_centroid))
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Create color gradient based on years
    num_years = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, num_years))
    
    # Create animation frames
    frames = []
    
    # Plot centroids with animation effect
    for i, (year, centroid) in enumerate(centroids):
        # Create new frame
        plt.clf()
        
        # Plot historical centroids with low alpha
        for j in range(i):
            prev_year, prev_centroid = centroids[j]
            plt.scatter(prev_centroid[0], prev_centroid[1], 
                       c=[colors[j]], s=100, alpha=0.2)
            plt.annotate(str(prev_year), (prev_centroid[0], prev_centroid[1]), 
                        xytext=(5, 5), textcoords='offset points', alpha=0.2)
        
        # Plot current centroid with full alpha
        plt.scatter(centroid[0], centroid[1], c=[colors[i]], s=100, alpha=0.8)
        plt.annotate(str(year), (centroid[0], centroid[1]), 
                    xytext=(5, 5), textcoords='offset points')
        
        # Draw line connecting to previous centroid if exists
        if i > 0:
            prev_centroid = centroids[i-1][1]
            plt.plot([prev_centroid[0], centroid[0]], 
                    [prev_centroid[1], centroid[1]], 
                    color=colors[i], alpha=0.5)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        title = "Evolution of Music Style Centroids Over Years"
        if artist_name:
            title = f"{artist_name}: {title}"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add colorbar to show year progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=min(years), vmax=max(years)))
        plt.colorbar(sm, label='Year')
        
        plt.tight_layout()
        
        # Save frame
        frames.append([plt.gca()])
    
    # Save animation
    anim = animation.ArtistAnimation(fig, frames, interval=500, blit=True)
    if artist_name:
        filename = f"centroid_evolution_{artist_name}.gif"
    else:
        filename = "centroid_evolution.gif"
    anim.save(filename, writer='pillow')
    plt.close()

# Example usage
#plot_centroid_evolution(cluster_results, sample_size=0.1)

# %%
def plot_centroid_movement(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """
    Plot the distance between consecutive year centroids over time
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Process data year by year
    years = sorted(cluster_data['year'].unique())
    centroids = []
    
    # Calculate centroid for each year
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            centroids.append((year, current_centroid))
    
    # Calculate distances between consecutive centroids
    distances = []
    movement_years = []
    
    for i in range(1, len(centroids)):
        prev_year, prev_centroid = centroids[i-1]
        curr_year, curr_centroid = centroids[i]
        
        distance = np.sqrt(np.sum((curr_centroid - prev_centroid) ** 2))
        distances.append(distance)
        movement_years.append(curr_year)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(movement_years, distances, '-o', linewidth=2, markersize=6)
    
    plt.xlabel('Year')
    plt.ylabel('Distance Between Consecutive Centroids')
    title = "Movement of Music Style Between Years"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='x', rotation=45)
    
    # Add mean distance line
    mean_distance = np.mean(distances)
    plt.axhline(y=mean_distance, color='r', linestyle='--', 
                label=f'Mean Distance: {mean_distance:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return movement_years, distances

# Example usage
plot_centroid_movement(cluster_results, sample_size=0.1)

# %%
def plot_movement_projection(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """
    Plot the projection of yearly movement vectors onto the overall movement direction
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Process data year by year to get centroids
    years = sorted(cluster_data['year'].unique())
    centroids = []
    
    for year in years:
        year_data = cluster_data[cluster_data['year'] == year]
        if not year_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in year_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            centroids.append((year, current_centroid))
    
    # Calculate overall movement vector (from first to last centroid)
    first_year, first_centroid = centroids[0]
    last_year, last_centroid = centroids[-1]
    overall_vector = last_centroid - first_centroid
    overall_direction = overall_vector / np.linalg.norm(overall_vector)
    
    # Calculate yearly movements and their projections
    projections = []
    movement_years = []
    
    for i in range(1, len(centroids)):
        prev_year, prev_centroid = centroids[i-1]
        curr_year, curr_centroid = centroids[i]
        
        # Calculate yearly movement vector
        yearly_vector = curr_centroid - prev_centroid
        
        # Calculate projection onto overall direction
        projection = np.dot(yearly_vector, overall_direction)
        projections.append(projection)
        movement_years.append(curr_year)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot projections
    plt.bar(movement_years, projections)
    
    plt.xlabel('Year')
    plt.ylabel('Projection Magnitude')
    title = "Contribution to Overall Music Style Movement"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    
    # Add zero line
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Add annotations
    plt.text(0.02, 0.98, 
            f'Positive values: Movement in overall direction\nNegative values: Movement against overall direction',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return movement_years, projections

# Example usage
plot_movement_projection(cluster_results, sample_size=0.1)

# %%
def analyze_pca_composition(features, feature_cols, k=None, threshold=0.9):
    """
    Analyze PCA components composition and their relationship with original features using Spark ML
    
    Args:
        features: Spark DataFrame with feature vectors
        feature_cols: List of original feature column names
        k: Optional fixed number of components
        threshold: Variance threshold for automatic component selection if k is None
        
    Returns:
        components_df: DataFrame showing composition of PCA components
    """
    # Get number of features
    n_features = features.first()["features"].size
    
    # Fit PCA model using Spark ML
    pca_spark = PCA(k=n_features, inputCol="features", outputCol="pcaFeatures")
    model_spark = pca_spark.fit(features)
    
    # Get components
    pc_matrix = model_spark.pc.toArray()  # Get principal components matrix
    
    # Use provided k or default to 2
    optimal_n = k if k is not None else 2
    
    # Create DataFrame with component compositions
    components_df = pd.DataFrame(
        pc_matrix[:, :optimal_n],
        columns=[f'PC{i+1}' for i in range(optimal_n)],
        index=feature_cols
    )
    
    # Plot component composition heatmap
    plt.figure(figsize=(6, 8))
    im = plt.imshow(components_df, cmap='RdBu', aspect='auto')
    plt.colorbar(im, label='Coefficient Value')
    
    # Add annotations
    for i in range(len(components_df.index)):
        for j in range(len(components_df.columns)):
            text = plt.text(j, i, f'{components_df.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    # Set ticks and labels
    plt.xticks(range(len(components_df.columns)), components_df.columns)
    plt.yticks(range(len(components_df.index)), components_df.index)
    
    plt.title('PCA Components Composition')
    plt.xlabel('Principal Components')
    plt.ylabel('Original Features')
    
    plt.tight_layout()
    plt.show()
    
    return components_df

# Example usage:
components_df = analyze_pca_composition(
    features, 
    feature_cols,
    k=2
)

# %%
