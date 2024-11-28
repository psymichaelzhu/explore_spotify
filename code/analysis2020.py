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
df = spark.read.csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True)

# 仅仅要release_date在1921-2020的
df = df.filter((F.col("release_date").between("1921-01-01", "2020-12-31")))
print(df.count())

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


# %% K-means and PCA visualization functions
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

    return optimal_n, pca_features, explained_variances, cumulative_variance, model


#%% run 
# 1. PCA: find optimal number of components
optimal_n, features_pca, explained_variances, cumulative_variance, model_pca = find_optimal_pca_components(features)
components_df = analyze_pca_composition(model_pca, feature_cols)
# 2. KMeans: find optimal k, based on PCA-transformed features
features_pca.persist()
optimal_k_pca, kmeans_predictions_pca, silhouettes_pca = find_optimal_kmeans(features_pca,k_values=range(2, 5))


#%% merge cluster results
# merge to get cluster results
merged_results = kmeans_predictions_pca.withColumn("tmp_id", F.monotonically_increasing_id()) \
            .join(df_features.withColumn("tmp_id", F.monotonically_increasing_id()).withColumnRenamed("features", "raw_features"), on="tmp_id", how="inner").drop("tmp_id") \
            .join(df,on=["id","name","artist"],how="inner")
merged_results.show()
merged_results.count()

# Get cluster counts
cluster_counts = merged_results.groupby('prediction').count()
cluster_counts.show()

# Filter out clusters with less than 100000 songs
small_clusters = cluster_counts.filter(F.col("count") < 100).select("prediction").rdd.flatMap(lambda x: x).collect()
cluster_results = merged_results.filter(~F.col("prediction").isin(small_clusters))

# filter out songs before 1920 and after 2020
cluster_results = cluster_results.filter(F.year(F.to_timestamp('release_date')).between(1920, 2020))

# old cluster distribution
cluster_results.groupby('prediction') \
              .count() \
              .orderBy('prediction') \
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
                np.sum((current_points - current_centroid) ** 2, axis=1))
            )
            current_distances_by_year[year] = current_distances
            
            # Calculate distances to historical centroid if available
            if historical_centroid is not None:
                historical_distances = np.sqrt(
                    np.sum((current_points - historical_centroid) ** 2, axis=1))
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
    
    # Filter data for 1920-2020
    cluster_data = cluster_data[
        (cluster_data['year'] >= 1920) & 
        (cluster_data['year'] <= 2020)
    ]
    
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
                        np.sum((current_points - historical_centroid) ** 2, axis=1))
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
                    np.sum((current_points - current_centroid) ** 2, axis=1))
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

'''
# 使用历史距离
results_df = plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.1,bins=[0,1,1.5,2,2.5,100])
plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.1,bins=[0,2,100])

# 用年距离
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.3,bins=[0,1,1.5,2,2.5,100])
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.3,bins=[0,2,100])

'''




# %% evolution of centroids
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



# %% plot yearly centroids movement vectors
def plot_yearly_movement_vectors(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """
    Plot yearly movement vectors with tails at origin to compare directions and magnitudes
    
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
    
    # Calculate yearly movement vectors
    movement_vectors = []
    movement_years = []
    
    for i in range(1, len(centroids)):
        prev_year, prev_centroid = centroids[i-1]
        curr_year, curr_centroid = centroids[i]
        
        # Calculate yearly movement vector
        yearly_vector = curr_centroid - prev_centroid
        movement_vectors.append(yearly_vector)
        movement_years.append(curr_year)
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Create color gradient based on years
    num_years = len(movement_years)
    colors = plt.cm.viridis(np.linspace(0, 1, num_years))
    
    # Plot movement vectors from origin
    for i, vector in enumerate(movement_vectors):
        plt.quiver(0, 0, vector[0], vector[1], 
                  angles='xy', scale_units='xy', scale=1,
                  color=colors[i], alpha=0.7)
        # Add year label at vector tip
        plt.annotate(str(movement_years[i]), 
                    (vector[0], vector[1]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add colorbar to show year progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min(movement_years), 
                                               vmax=max(movement_years)))
    plt.colorbar(sm, label='Year')
    
    plt.xlabel('First Principal Component Change')
    plt.ylabel('Second Principal Component Change')
    title = "Yearly Movement Vectors in Music Style Space"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_yearly_movement_vectors(cluster_results, sample_size=0.1)

plot_yearly_movement_vectors(cluster_results, artist_name="Taylor Swift",sample_size=0.1)

# %% evolution of music (more than centroids)
def plot_music_era_heatmap(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Plot a heatmap showing the distribution of music eras across the PCA space
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata
        sample_size: Fraction of data to sample
        grid_size: Number of grid cells in each dimension
        seed: Random seed for sampling
    """
    # Sample and convert to pandas
    _, _, df = exact_to_pd(cluster_data, sample_size=sample_size, seed=seed)
    
    # Extract PCA coordinates and years
    pca_coords = np.vstack(df['features'].values)
    years = pd.to_datetime(df['release_date']).dt.year.values
    
    # Create 2D histogram grid
    x_edges = np.linspace(pca_coords[:,0].min(), pca_coords[:,0].max(), grid_size)
    y_edges = np.linspace(pca_coords[:,1].min(), pca_coords[:,1].max(), grid_size)
    
    # Initialize grid for storing average years
    year_grid = np.zeros((grid_size-1, grid_size-1))
    count_grid = np.zeros((grid_size-1, grid_size-1))
    
    # Calculate average year for each grid cell
    for x, y, year in zip(pca_coords[:,0], pca_coords[:,1], years):
        x_idx = np.digitize(x, x_edges) - 1
        y_idx = np.digitize(y, y_edges) - 1
        if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
            year_grid[y_idx, x_idx] += year
            count_grid[y_idx, x_idx] += 1
    
    # Calculate average years, handling division by zero
    mask = count_grid > 0
    year_grid[mask] = year_grid[mask] / count_grid[mask]
    year_grid[~mask] = np.nan
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap using viridis colormap
    im = plt.imshow(year_grid, 
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    origin='lower',
                    cmap='viridis',
                    aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Year')
    
    # Customize plot
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Distribution of Music Eras in PCA Space')
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_music_era_heatmap(cluster_results, sample_size=0.3, grid_size=30)

# %% plot music feature trends
def plot_music_trends(cluster_data, sample_size=0.1, seed=None, window_size=3, milestone_years=None):
    """
    Plot trends of different musical features over time using smoothed curves in separate facets
    
    Args:
        cluster_data: Clustered data containing musical features
        sample_size: Fraction of data to sample (default: 0.1)
        seed: Random seed for sampling (default: None)
        window_size: Size of rolling window for smoothing (default: 3)
        milestone_years: List of years to mark with vertical lines (default: None)
    """
    # Sample and prepare data
    _, _, df = exact_to_pd(cluster_data, sample_size=sample_size, seed=seed)
    
    # Features to analyze
    features = ['acousticness', 'instrumentalness','mode', 
    'danceability', 'valence', 'tempo', 
    'energy', 'time_signature','loudness', 
    'duration_ms','key','speechiness']
    
    # Calculate yearly averages, se and normalize
    yearly_trends = pd.DataFrame()
    yearly_ses = pd.DataFrame()
    for feature in features:
        # Get feature values from raw_features array based on original order
        feature_idx = feature_cols.index(feature)
        df[feature] = df['raw_features'].apply(lambda x: x[feature_idx])
        
        # Calculate yearly means and standard errors
        yearly_stats = df.groupby('year')[feature].agg(['mean', 'std', 'count'])
        yearly_mean = yearly_stats['mean']
        yearly_se = yearly_stats['std'] / np.sqrt(yearly_stats['count'])
        
        # Normalize to 0-1 scale
        mean_min, mean_max = yearly_mean.min(), yearly_mean.max()
        yearly_normalized = (yearly_mean - mean_min) / (mean_max - mean_min)
        yearly_se_normalized = yearly_se / (mean_max - mean_min)
        
        # Apply smoothing using rolling average
        yearly_trends[feature] = yearly_normalized.rolling(window=window_size, center=True).mean()
        yearly_ses[feature] = yearly_se_normalized.rolling(window=window_size, center=True).mean()

    # Create subplots grid
    n_rows = 4
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    axes = axes.flatten()
    
    # Plot each feature in its own subplot
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Plot mean line
        ax.plot(yearly_trends.index, yearly_trends[feature], 
                color='blue',
                linewidth=2,
                label='Mean')
        
        # Add confidence interval
        ax.fill_between(yearly_trends.index,
                       yearly_trends[feature] - yearly_ses[feature],
                       yearly_trends[feature] + yearly_ses[feature],
                       color='blue',
                       alpha=0.2,
                       label='±1 SE')
        
        # Add vertical lines for milestone years if provided
        if milestone_years is not None:
            for year in milestone_years:
                if year >= yearly_trends.index.min() and year <= yearly_trends.index.max():
                    ax.axvline(x=year, color='red', linestyle='--', alpha=0.5, 
                             label=f'Milestone {year}')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Normalized Score')
        ax.set_title(feature.capitalize())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Musical Feature Trends Over Time', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage
plot_music_trends(cluster_results, sample_size=0.1, window_size=14, milestone_years=[1950, 1985])




# %% time series analysis: function
def time_series_analysis(cluster_data, artist_name=None, sample_size=0.1, seed=None,
                                   bins=[0, 100], distance_type='historical',
                                   forecast_years=0, cycle_years=40, holidays=['1966-01-01', '1993-01-01']):
    """
    Analyze innovation levels using Prophet with holidays and custom regressors for streaming periods.
    
    Args:
        cluster_data: DataFrame with cluster predictions and features.
        artist_name: Optional artist name to filter data.
        sample_size: Fraction of data to sample.
        seed: Random seed for sampling.
        bins: Innovation level ranges.
        distance_type: Type of distance to calculate ('historical' or 'internal').
        forecast_years: Number of years to forecast.
        cycle_years: Length of cycle in years for custom seasonality.
        holidays: List of dates for holidays or special events.
    """
    from prophet import Prophet
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Step 1: Prepare innovation data
    results_df = plot_innovation_levels_over_time(
        cluster_data, artist_name, sample_size, seed, bins, distance_type
    )
    
    # Step 2: Define holidays if provided
    holidays_df = None
    if holidays:
        holidays_df = pd.DataFrame({
            'holiday': 'innovation_peak',
            'ds': pd.to_datetime(holidays),
            'lower_window': -2,
            'upper_window': 2,
        })
    
    prophet_results = {}
    bin_ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    
    for range_name in bin_ranges:
        # Step 3: Create DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(results_df.index.astype(str) + '-01-01'),
            'y': results_df[range_name]
        })
        
        # Add regressors for streaming events
        df['year_since_streaming'] = (df['ds'].dt.year - 1999).clip(lower=0)
        df['streaming'] = (df['ds'] >= '1999-01-01').astype(int)
        
        # Step 4: Initialize Prophet model
        model = Prophet(
            mcmc_samples=200,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='additive',
            changepoint_prior_scale=0.001,
            holidays_prior_scale=10.0,
        )
        
        # Add custom seasonality
        model.add_seasonality(
            name='long_cycle',
            period=cycle_years * 365.25,
            fourier_order=3,
            prior_scale=1.0
        )
        
        # Add regressors for intercept and slope changes
        model.add_regressor('streaming', prior_scale=10.0)
        model.add_regressor('year_since_streaming', prior_scale=10.0)
        
        # Step 5: Fit the model
        model.fit(df)
        
        # Step 6: Make future predictions
        future = model.make_future_dataframe(periods=forecast_years, freq='Y')
        future['year_since_streaming'] = (future['ds'].dt.year - 1999).clip(lower=0)
        future['streaming'] = (future['ds'] >= '1999-01-01').astype(int)
        
        forecast = model.predict(future)
        
        # Step 7: Store results
        prophet_results[range_name] = {
            'model': model,
            'forecast': forecast,
            'actual': df
        }
        
        # Step 8: Visualization
        
        # Plot components
        model.plot_components(forecast)
        plt.suptitle(f'Analysis for Innovation Range {range_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Plot forecast with regressors
        plt.figure(figsize=(12, 6))
        fig = model.plot(forecast)
        ax = fig.gca()
        plt.title(f'Innovation Level Forecast for Range {range_name}\n(Including Regressor Effects)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    return prophet_results

def visualize_fit_and_jointplot(prophet_results, bins=None, figsize=(10, 8)):
    """
    Visualize the fit of Prophet models and create joint plots of predicted vs observed data.

    Args:
        prophet_results: Dictionary containing Prophet models and results (output of time_series_analysis).
        bins: Optional list of bin ranges for filtering specific results.
        figsize: Tuple specifying the figure size for plots.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Iterate through each range and visualize
    for range_name, result in prophet_results.items():
        # If bins are specified, filter by range name
        if bins and range_name not in bins:
            continue
        
        model = result['model']
        forecast = result['forecast']
        actual = result['actual']
        
        # Merge forecast and actual data for comparison˛
        merged = forecast[['ds', 'yhat']].merge(actual[['ds', 'y']], on='ds', how='inner')
        
        # Compute residuals
        merged['residual'] = merged['y'] - merged['yhat']
        
        # Create jointplot for predicted vs observed values
        g = sns.jointplot(
            data=merged,
            x='yhat',
            y='y',
            kind='reg',
            height=8,
            marginal_kws=dict(bins=30, fill=True),
            joint_kws=dict(scatter_kws={'alpha': 0.5}),
        )
        
        # Add correlation and MAE to the plot
        r2 = np.corrcoef(merged['yhat'], merged['y'])[0, 1]**2
        mae = np.mean(np.abs(merged['residual']))
        g.ax_joint.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
                        transform=g.ax_joint.transAxes, verticalalignment='top')
        
        # Titles and labels
        g.fig.suptitle(f"Joint Plot for Range: {range_name}", y=1.02)
        g.set_axis_labels("Predicted (yhat)", "Observed (y)")
        plt.show()

        # Plot residuals
        plt.figure(figsize=figsize)
        sns.histplot(merged['residual'], kde=True, bins=30, color='#3C76AF')
        plt.title(f"Residual Distribution for Range: {range_name}", fontsize=14)
        plt.xlabel("Residuals (Observed - Predicted)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Plot time series fit
        plt.figure(figsize=figsize)
        plt.scatter(actual['ds'], actual['y'], label='Observed', color='black',alpha=0.8, s=20)
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='#3C76AF',linewidth=2)
        plt.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color='#3a76AF', alpha=0.2, label='Prediction Interval'
        )
        plt.title(f"Time Series Fit for Range: {range_name}", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.legend()
        plt.tight_layout()
        plt.show()

def generate_new_df(prophet_results, cluster_data, artist_name=None, sample_size=0.1, seed=None,
                                   bins=[0, 100], distance_type='historical',
                                   forecast_years=0, cycle_years=40, holidays=['1966-01-01', '1993-01-01']):
    """
    Analyze innovation levels using Prophet with holidays and custom regressors for streaming periods.
    
    Args:
        cluster_data: DataFrame with cluster predictions and features.
        artist_name: Optional artist name to filter data.
        sample_size: Fraction of data to sample.
        seed: Random seed for sampling.
        bins: Innovation level ranges.
        distance_type: Type of distance to calculate ('historical' or 'internal').
        forecast_years: Number of years to forecast.
        cycle_years: Length of cycle in years for custom seasonality.
        holidays: List of dates for holidays or special events.
    """
    import copy
    # Step 1: Prepare innovation data
    results_df = plot_innovation_levels_over_time(
        cluster_data, artist_name, sample_size, seed, bins, distance_type
    )
    bin_ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    new_results = copy.deepcopy(prophet_results)
    for range_name in bin_ranges:
        # Step 3: Create DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(results_df.index.astype(str) + '-01-01'),
            'y': results_df[range_name]
        })
        
        # Add regressors for streaming events
        df['year_since_streaming'] = (df['ds'].dt.year - 1999).clip(lower=0)
        df['streaming'] = (df['ds'] >= '1999-01-01').astype(int)
        
        new_results[range_name]['actual'] = df

    
    return new_results


#%% cross-validation: 滚动CV
splits = cluster_results.randomSplit([0.5, 0.5], seed=42)
cluster_results_1 = splits[0]
cluster_results_2 = splits[1]
sample_size = 0.2
distance_type = 'internal'
prophet_results = time_series_analysis(
    cluster_results_1, 
    distance_type=distance_type,#historical, internal
    sample_size=sample_size,
    bins=[0,2,100],#0,0.5,1, 1.5, 2, 2.5, 100
    forecast_years=0,
    cycle_years=40,
    holidays=['1966-01-01', '1993-01-01']
)
new_results = generate_new_df(
    prophet_results, 
    cluster_results_2, 
    distance_type=distance_type,#historical, internal
    sample_size=sample_size,
    bins=[0,2,100],#0,0.5,1, 1.5, 2, 2.5, 100
    forecast_years=0,
    cycle_years=40,
    holidays=['1966-01-01', '1993-01-01']
)
visualize_fit_and_jointplot(prophet_results, bins=["2.0-100.0","0.0-2.0"])
visualize_fit_and_jointplot(new_results, bins=["2.0-100.0","0.0-2.0"])

# Extract actual values from both results for 2.0-100.0
actual_values_1_high = prophet_results["2.0-100.0"]["actual"]
actual_values_2_high = new_results["2.0-100.0"]["actual"]

# Extract actual values for 0.0-2.0
actual_values_1_low = prophet_results["0.0-2.0"]["actual"]
actual_values_2_low = new_results["0.0-2.0"]["actual"]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot 2.0-100.0 range
ax1.plot(actual_values_1_high['ds'], actual_values_1_high['y'], label='Training Set', marker='o')
ax1.plot(actual_values_2_high['ds'], actual_values_2_high['y'], label='Test Set', marker='o')
ax1.set_title('Comparison of Training and Test Sets (2.0-100.0)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True)

# Plot 0.0-2.0 range
ax2.plot(actual_values_1_low['ds'], actual_values_1_low['y'], label='Training Set', marker='o')
ax2.plot(actual_values_2_low['ds'], actual_values_2_low['y'], label='Test Set', marker='o')
ax2.set_title('Comparison of Training and Test Sets (0.0-2.0)')
ax2.set_xlabel('Year')
ax2.set_ylabel('Value')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()





# %% coefficient visualization
import numpy as np
from prophet.utilities import regressor_coefficients
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_and_visualize_regressor_coefficients_with_ci(prophet_results, regressors, bins=None):
    """
    Extract and visualize regressor coefficients with credible intervals (CI) from Prophet models.
    
    Args:
        prophet_results: Dictionary containing Prophet models and results.
        regressors: List of regressors to analyze.
        bins: Optional list of range names to include in the analysis.
    """
    coefficients = []

    for range_name, result in prophet_results.items():
        if bins and range_name not in bins:
            continue

        # Extract coefficients and credible intervals
        model = result['model']
        regressor_data = regressor_coefficients(model)
        
        if 'coef' in regressor_data.columns:
            for _, row in regressor_data.iterrows():
                if row['regressor'] in regressors:
                    coefficients.append({
                        'range': 'Low' if range_name == '0.0-2.0' else 'High',
                        'regressor': 'intercept' if row['regressor'] == 'streaming' else 'slope',
                        'coefficient': row['coef'],
                        'lower_bound': row['coef_lower'] if 'coef_lower' in row else np.nan,
                        'upper_bound': row['coef_upper'] if 'coef_upper' in row else np.nan
                    })

    coefficients_df = pd.DataFrame(coefficients)
    print(coefficients_df)
    
    # Create a figure for coefficient visualization
    plt.figure(figsize=(4, 6))

    # Define colors for each range
    range_colors = {
        "High": "#4169E1",  # Royal blue with lower saturation
        "Low": "#CD5C5C"     # Indian red with lower saturation
    }

    # Plot coefficients with error bars
    for i, regressor in enumerate(['intercept', 'slope']):
        regressor_data = coefficients_df[coefficients_df['regressor'] == regressor]
        
        # Calculate x positions for each range, offset by regressor
        x_positions = np.arange(len(regressor_data)) + i * 0.45
        
        for j, (idx, row) in enumerate(regressor_data.iterrows()):
            plt.errorbar(x_positions[j], 
                        row['coefficient'],
                        yerr=[[row['coefficient'] - row['lower_bound']],
                              [row['upper_bound'] - row['coefficient']]],
                        fmt='o',
                        capsize=5,
                        capthick=2,
                        label=f"{regressor} ({row['range']})",
                        markersize=8,
                        color=range_colors[row['range']])

    # Customize plot
    plt.xticks(np.arange(len(bins)) + 0.175, ['Low', 'High'])
    plt.xlabel('Innovation Level')
    plt.ylabel('Coefficient Value')
    plt.title('Regressor Coefficients with 95% Credible Intervals')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example Usage
extract_and_visualize_regressor_coefficients_with_ci(
    prophet_results,
    regressors=['streaming', 'year_since_streaming'],
    bins=["2.0-100.0", "0.0-2.0"]
)


#%% artist similarity trends
def analyze_artist_similarity_trends_pandas(cluster_data, sample_size=0.1, distance_type='internal'):
    """
    Analyze artist similarity trends using pandas sampling approach
    
    Args:
        cluster_data: PySpark DataFrame with features and metadata
        sample_size: Fraction of artists to sample (default 0.1)
        distance_type: Type of similarity to calculate ('internal', 'historical', or 'between')
    """
    # Get list of unique artists and sample
    unique_artists = cluster_data.select('artist').distinct()
    sampled_artists = unique_artists.sample(withReplacement=False, fraction=sample_size)
    
    # Filter data for sampled artists and convert to pandas
    pdf = cluster_data.join(sampled_artists, 'artist').toPandas()
    
    # Extract year from release_date
    pdf['year'] = pd.to_datetime(pdf['release_date']).dt.year
    
    # Filter years
    pdf = pdf[(pdf['year'] >= 1920) & (pdf['year'] <= 2020)]
    
    years = sorted(pdf['year'].unique())
    similarity_by_year = {'year': [], 'avg_similarity': [], 'std_similarity': []}
    historical_centroid = None
    
    for year in years:
        year_data = pdf[pdf['year'] == year]
        
        if len(year_data) > 0:
            if distance_type == 'internal':
                # Calculate within-artist similarity
                artist_similarities = []
                for artist in year_data['artist'].unique():
                    artist_songs = year_data[year_data['artist'] == artist]
                    if len(artist_songs) > 1:
                        artist_features = np.array([np.array(f) for f in artist_songs['features']])
                        artist_centroid = artist_features.mean(axis=0)
                        distances = np.sqrt(((artist_features - artist_centroid) ** 2).sum(axis=1))
                        artist_similarities.extend(distances)
                
                if artist_similarities:
                    similarity_by_year['year'].append(year)
                    similarity_by_year['avg_similarity'].append(np.mean(artist_similarities))
                    similarity_by_year['std_similarity'].append(np.std(artist_similarities))
                    
            elif distance_type == 'historical':
                # Calculate historical similarity per artist
                artist_similarities = []
                for artist in year_data['artist'].unique():
                    artist_data = pdf[pdf['artist'] == artist]
                    artist_historical = artist_data[artist_data['year'] <= year]
                    
                    if not artist_historical.empty:
                        if historical_centroid is None:
                            historical_features = np.array([np.array(f) for f in artist_historical['features']])
                            historical_centroid = historical_features.mean(axis=0)
                            
                        current_songs = artist_data[artist_data['year'] == year]
                        current_features = np.array([np.array(f) for f in current_songs['features']])
                        distances = np.sqrt(((current_features - historical_centroid) ** 2).sum(axis=1))
                        artist_similarities.extend(distances)
                        
                        # Update historical centroid for this artist
                        historical_features = np.array([np.array(f) for f in artist_historical['features']])
                        historical_centroid = historical_features.mean(axis=0)
                
                if artist_similarities:
                    similarity_by_year['year'].append(year)
                    similarity_by_year['avg_similarity'].append(np.mean(artist_similarities))
                    similarity_by_year['std_similarity'].append(np.std(artist_similarities))
                
            else:  # between artists
                # Calculate between-artist similarity
                artist_centroids = {}
                for artist in year_data['artist'].unique():
                    artist_songs = year_data[year_data['artist'] == artist]
                    artist_features = np.array([np.array(f) for f in artist_songs['features']])
                    artist_centroids[artist] = artist_features.mean(axis=0)
                
                if len(artist_centroids) > 1:
                    # Calculate pairwise distances between centroids
                    distances = []
                    artists = list(artist_centroids.keys())
                    for i in range(len(artists)):
                        for j in range(i + 1, len(artists)):
                            dist = np.sqrt(((artist_centroids[artists[i]] - artist_centroids[artists[j]]) ** 2).sum()
                            distances.append(dist)
                            
                    similarity_by_year['year'].append(year)
                    similarity_by_year['avg_similarity'].append(np.mean(distances))
                    similarity_by_year['std_similarity'].append(np.std(distances))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    data = pd.DataFrame(similarity_by_year)
    plt.plot(data['year'], data['avg_similarity'], label=f'{distance_type.capitalize()} Similarity', marker='o')
    plt.fill_between(
        data['year'],
        data['avg_similarity'] - data['std_similarity'],
        data['avg_similarity'] + data['std_similarity'],
        alpha=0.2
    )
    plt.xlabel('Year')
    plt.ylabel('Similarity Score')
    plt.title(f'{distance_type.capitalize()} Artist Similarity Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
analyze_artist_similarity_trends_pandas(cluster_results, sample_size=0.1, distance_type='internal')
analyze_artist_similarity_trends_pandas(cluster_results, sample_size=0.1, distance_type='between')


# %%
# %% evolution of music (yearly distribution)
def plot_yearly_distribution(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Plot the distribution of songs in PCA space for each year
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata
        sample_size: Fraction of data to sample
        grid_size: Number of grid cells in each dimension
        seed: Random seed for sampling
    """
    # Sample and convert to pandas
    _, _, df = exact_to_pd(cluster_data, sample_size=sample_size, seed=seed)
    
    # Extract PCA coordinates and years
    pca_coords = np.vstack(df['features'].values)
    years = pd.to_datetime(df['release_date']).dt.year.values
    
    # Create grid for density estimation
    x_edges = np.linspace(pca_coords[:,0].min(), pca_coords[:,0].max(), grid_size)
    y_edges = np.linspace(pca_coords[:,1].min(), pca_coords[:,1].max(), grid_size)
    
    # Create figure with subplots
    unique_years = sorted(np.unique(years))
    n_years = len(unique_years)
    n_cols = 5
    n_rows = (n_years + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    # Plot distribution for each year
    for idx, year in enumerate(unique_years):
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
            im = axes[idx].imshow(
                hist.T,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin='lower',
                cmap='viridis',
                aspect='auto'
            )
            
            axes[idx].set_title(f'Year: {year}')
            axes[idx].set_xlabel('First Principal Component')
            axes[idx].set_ylabel('Second Principal Component')
    
    # Remove empty subplots
    for idx in range(n_years, len(axes)):
        fig.delaxes(axes[idx])
    
    # Add colorbar
    plt.colorbar(im, ax=axes, label='Number of Songs')
    
    plt.suptitle('Evolution of Music Distribution in PCA Space', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage
plot_yearly_distribution(cluster_results, sample_size=0.1)

# %% animation
def plot_yearly_distribution_animation(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Create an animated plot showing the evolution of song distribution in PCA space over time
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata
        sample_size: Fraction of data to sample
        grid_size: Number of grid cells in each dimension
        seed: Random seed for sampling
    """
    # Sample and convert to pandas
    _, _, df = exact_to_pd(cluster_data, sample_size=sample_size, seed=seed)
    
    # Extract PCA coordinates and years
    pca_coords = np.vstack(df['features'].values)
    years = pd.to_datetime(df['release_date']).dt.year.values
    
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
    plt.tight_layout()
    
    # Save animation
    anim.save('music_distribution_evolution.gif', writer='pillow')
    plt.show()

# Example usage
plot_yearly_distribution_animation(cluster_results, sample_size=0.1)

# %% pca animation
def plot_pca_distribution_animation(cluster_data, sample_size=0.1, seed=42):
    """
    Create an animated plot showing the distribution of songs along PCA components over time
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
    """
    from scipy.stats import gaussian_kde
    
    try:
        # Sample and convert to pandas
        _, _, df = exact_to_pd(cluster_data, sample_size=sample_size, seed=seed)
        
        # Extract PCA coordinates and years
        pca_coords = np.vstack(df['features'].values)
        years = pd.to_datetime(df['release_date']).dt.year.values
        
        # Create figure with two subplots for each component
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Distribution of Songs Along Principal Components', fontsize=14)
        
        # Set up plot elements
        line1, = ax1.plot([], [], lw=2)
        line2, = ax2.plot([], [], lw=2)
        
        # Set axis labels
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Density')
        ax2.set_xlabel('Second Principal Component')
        ax2.set_ylabel('Density')
        
        # Set axis limits based on data
        x1_min, x1_max = pca_coords[:,0].min(), pca_coords[:,0].max()
        x2_min, x2_max = pca_coords[:,1].min(), pca_coords[:,1].max()
        
        ax1.set_xlim(x1_min, x1_max)
        ax2.set_xlim(x2_min, x2_max)
        
        # Calculate max density across all years to set fixed y limits
        max_density1 = 0
        max_density2 = 0
        for year in np.unique(years):
            year_mask = years == year
            year_coords = pca_coords[year_mask]
            if len(year_coords) > 0:
                kde1 = gaussian_kde(year_coords[:,0])
                kde2 = gaussian_kde(year_coords[:,1])
                x1 = np.linspace(x1_min, x1_max, 200)
                x2 = np.linspace(x2_min, x2_max, 200)
                max_density1 = max(max_density1, max(kde1(x1)))
                max_density2 = max(max_density2, max(kde2(x2)))
        
        # Set fixed y limits
        ax1.set_ylim(0, max_density1 * 1.1)
        ax2.set_ylim(0, max_density2 * 1.1)
        
        # Add year text
        year_text = fig.text(0.02, 0.98, '', fontsize=12)
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2
        
        def update(frame):
            year = sorted(np.unique(years))[frame]
            year_mask = years == year
            year_coords = pca_coords[year_mask]
            
            if len(year_coords) > 0:
                # Calculate KDE for first component
                kde1 = gaussian_kde(year_coords[:,0])
                x1 = np.linspace(x1_min, x1_max, 200)
                y1 = kde1(x1)
                line1.set_data(x1, y1)
                
                # Calculate KDE for second component
                kde2 = gaussian_kde(year_coords[:,1])
                x2 = np.linspace(x2_min, x2_max, 200)
                y2 = kde2(x2)
                line2.set_data(x2, y2)
                
                # Update year text
                year_text.set_text(f'Year: {year}')
            
            return line1, line2
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, 
            update,
            init_func=init,
            frames=len(np.unique(years)),
            interval=100,
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        
        # Save animation
        anim.save('pca_distribution_evolution.gif', writer='pillow')
        plt.show()
    except Exception as e:
        print(f"Error creating PCA distribution animation: {e}")

# Example usage
plot_pca_distribution_animation(cluster_results, sample_size=0.1)

# %% pca animation 5 year
def plot_pca_distribution_animation(cluster_data, sample_size=0.1, seed=42, year_window=5):
    """
    Create an animated plot showing the distribution of songs along PCA components over time,
    averaged over 5-year windows
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
        year_window: Size of year window for averaging (default 5)
    """
    from scipy.stats import gaussian_kde
    
    try:
        # Sample and convert to pandas
        _, _, df = exact_to_pd(cluster_data, sample_size=sample_size, seed=seed)
        
        # Extract PCA coordinates and years
        pca_coords = np.vstack(df['features'].values)
        years = pd.to_datetime(df['release_date']).dt.year.values
        
        # Create year bins
        min_year = (years.min() // year_window) * year_window
        max_year = ((years.max() // year_window) + 1) * year_window
        year_bins = np.arange(min_year, max_year + year_window, year_window)
        
        # Create figure with two subplots for each component
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Distribution of Songs Along Principal Components (5-Year Averages)', fontsize=14)
        
        # Set up plot elements
        line1, = ax1.plot([], [], lw=2)
        line2, = ax2.plot([], [], lw=2)
        
        # Set axis labels
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Density')
        ax2.set_xlabel('Second Principal Component')
        ax2.set_ylabel('Density')
        
        # Set axis limits based on data
        x1_min, x1_max = pca_coords[:,0].min(), pca_coords[:,0].max()
        x2_min, x2_max = pca_coords[:,1].min(), pca_coords[:,1].max()
        
        ax1.set_xlim(x1_min, x1_max)
        ax2.set_xlim(x2_min, x2_max)
        
        # Calculate max density across all year windows to set fixed y limits
        max_density1 = 0
        max_density2 = 0
        for start_year in year_bins[:-1]:
            year_mask = (years >= start_year) & (years < start_year + year_window)
            year_coords = pca_coords[year_mask]
            if len(year_coords) > 0:
                kde1 = gaussian_kde(year_coords[:,0])
                kde2 = gaussian_kde(year_coords[:,1])
                x1 = np.linspace(x1_min, x1_max, 200)
                x2 = np.linspace(x2_min, x2_max, 200)
                max_density1 = max(max_density1, max(kde1(x1)))
                max_density2 = max(max_density2, max(kde2(x2)))
        
        # Set fixed y limits
        ax1.set_ylim(0, max_density1 * 1.1)
        ax2.set_ylim(0, max_density2 * 1.1)
        
        # Add year range text
        year_text = fig.text(0.02, 0.98, '', fontsize=12)
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2
        
        def update(frame):
            start_year = year_bins[frame]
            year_mask = (years >= start_year) & (years < start_year + year_window)
            year_coords = pca_coords[year_mask]
            
            if len(year_coords) > 0:
                # Calculate KDE for first component
                kde1 = gaussian_kde(year_coords[:,0])
                x1 = np.linspace(x1_min, x1_max, 200)
                y1 = kde1(x1)
                line1.set_data(x1, y1)
                
                # Calculate KDE for second component
                kde2 = gaussian_kde(year_coords[:,1])
                x2 = np.linspace(x2_min, x2_max, 200)
                y2 = kde2(x2)
                line2.set_data(x2, y2)
                
                # Update year text
                year_text.set_text(f'Years: {start_year}-{start_year + year_window - 1}')
            
            return line1, line2
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, 
            update,
            init_func=init,
            frames=len(year_bins)-1,
            interval=80,  # Increased interval for better viewing
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        
        # Save animation
        anim.save('pca_distribution_evolution_5year.gif', writer='pillow')
        plt.show()
    except Exception as e:
        print(f"Error creating PCA distribution animation: {e}")

# Example usage
plot_pca_distribution_animation(cluster_results, sample_size=0.1)


# %% centroid
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType, ArrayType

def compute_centroid_from_vector(df, feature_col='features'):
    """
    Compute the centroid of the features vector column.

    Args:
        df: Input DataFrame containing a 'features' column of type Vector.
        feature_col: Name of the vector column.

    Returns:
        A list representing the centroid vector.
    """
    # Step 1: Convert vector to array
    df_with_array = df.withColumn('features_array', vector_to_array(F.col(feature_col)))
    
    # Step 2: Get the number of dimensions in the vector
    num_features = df_with_array.select(F.size('features_array').alias('num_features')).first()['num_features']
    
    # Step 3: Extract each dimension into a separate column
    for i in range(num_features):
        df_with_array = df_with_array.withColumn(f'feature_{i}', F.col('features_array')[i])
    
    # Step 4: Compute the average for each dimension
    avg_features = df_with_array.agg(
        *[F.avg(F.col(f'feature_{i}')).alias(f'feature_{i}') for i in range(num_features)]
    ).collect()[0]
    
    # Step 5: Convert back to a list or Vector (optional)
    centroid = [avg_features[f'feature_{i}'] for i in range(num_features)]
    
    return centroid

# 示例用法
centroid = compute_centroid_from_vector(cluster_results, feature_col='features')
print("Centroid:", centroid)




# %% similarity trends
# similarity between and within artists
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
import matplotlib.pyplot as plt
import pandas as pd
def calculate_yearly_dispersion(cluster_data, sample_size=0.1, seed=42):
    """
    Calculate yearly dispersion of songs from their centroids in PCA space
    
    Args:
        cluster_data: PySpark DataFrame with features and metadata
        sample_size: Fraction of songs to sample (default 0.1)
        seed: Random seed for sampling
    
    Returns:
        Pandas DataFrame with yearly dispersion metrics
    """
    # Sample data
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Extract year from release_date
    sampled_data = sampled_data.withColumn(
        'year',
        F.year(F.to_timestamp('release_date'))
    )
    
    # Convert features to array
    sampled_data = sampled_data.withColumn('features_array', vector_to_array(F.col('features')))
    
    # Get number of dimensions
    num_features = sampled_data.select(F.size('features_array').alias('num_features')).first()['num_features']
    
    # Extract each dimension into separate columns
    for i in range(num_features):
        sampled_data = sampled_data.withColumn(f'feature_{i}', F.col('features_array')[i])
    
    # Calculate yearly centroids
    yearly_centroids = sampled_data.groupBy('year').agg(
        *[F.avg(F.col(f'feature_{i}')).alias(f'centroid_{i}') for i in range(num_features)]
    )
    
    # Join centroids back to main data
    data_with_centroids = sampled_data.join(yearly_centroids, on='year')
    
    # Calculate distances to centroids
    distance_calc = '+'.join([f'pow(feature_{i} - centroid_{i}, 2)' for i in range(num_features)])
    data_with_distances = data_with_centroids.withColumn(
        'distance_to_centroid',
        F.sqrt(F.expr(distance_calc))
    )
    
    # Calculate yearly statistics
    yearly_stats = data_with_distances.groupBy('year').agg(
        F.avg('distance_to_centroid').alias('avg_distance'),
        F.stddev('distance_to_centroid').alias('std_distance'),
        F.expr('stddev(distance_to_centroid) / sqrt(count(*))').alias('se_distance'),
        F.count('*').alias('song_count')
    )
    
    # Convert to pandas for visualization
    yearly_stats_pd = yearly_stats.toPandas()
    yearly_stats_pd = yearly_stats_pd.sort_values('year')
    
    return yearly_stats_pd

def compute_artist_similarity(cluster_data, distance_type='within', sample_size=0.1, seed=42):
    """
    Compute artist-level similarity based on the features vector.

    Args:
        cluster_data: PySpark DataFrame with features and metadata.
        distance_type: Type of similarity to compute ('within' or 'between').
        sample_size: Fraction of data to sample for efficiency (default 0.1).
        seed: Random seed for sampling (default 42).

    Returns:
        Pandas DataFrame with similarity metrics over time.
    """
    # Step 1: Sample data
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Step 2: Extract year from release_date
    sampled_data = sampled_data.withColumn(
        'year',
        F.year(F.to_timestamp('release_date'))
    )
    
    # Step 3: Convert features to array
    sampled_data = sampled_data.withColumn('features_array', vector_to_array(F.col('features')))
    
    # Step 4: Determine number of dimensions
    num_features = sampled_data.select(F.size('features_array').alias('num_features')).first()['num_features']
    
    # Step 5: Extract each feature dimension into separate columns
    for i in range(num_features):
        sampled_data = sampled_data.withColumn(f'feature_{i}', F.col('features_array')[i])
    
    if distance_type == 'within':
        # Step 6a: Calculate centroids for each artist-year
        artist_centroids = sampled_data.groupBy('artist', 'year').agg(
            *[F.avg(F.col(f'feature_{i}')).alias(f'centroid_{i}') for i in range(num_features)]
        )
        
        # Step 7a: Join centroids back to original data
        data_with_centroids = sampled_data.join(artist_centroids, on=['artist', 'year'])
        
        # Step 8a: Compute distances to centroids
        distance_calc = '+'.join([f'pow(feature_{i} - centroid_{i}, 2)' for i in range(num_features)])
        data_with_distances = data_with_centroids.withColumn(
            'distance_to_centroid',
            F.sqrt(F.expr(distance_calc))
        )
        
        # Step 9a: Calculate within-artist similarity per year
        yearly_stats = data_with_distances.groupBy('year').agg(
            F.avg('distance_to_centroid').alias('avg_distance'),
            F.stddev('distance_to_centroid').alias('std_distance'),
            F.expr('stddev(distance_to_centroid) / sqrt(count(*))').alias('se_distance'),
            F.count('*').alias('song_count')
        )
        
    elif distance_type == 'between':
        # Step 6b: Calculate centroids for each artist-year
        artist_centroids = sampled_data.groupBy('artist', 'year').agg(
            *[F.avg(F.col(f'feature_{i}')).alias(f'centroid_{i}') for i in range(num_features)]
        )
        
        # Step 7b: Calculate yearly centroids across all artists
        yearly_centroids = artist_centroids.groupBy('year').agg(
            *[F.avg(F.col(f'centroid_{i}')).alias(f'global_centroid_{i}') for i in range(num_features)]
        )
        
        # Step 8b: Join yearly centroids to artist centroids
        data_with_centroids = artist_centroids.join(yearly_centroids, on='year')
        
        # Step 9b: Compute distances between artist centroids and global centroid
        distance_calc = '+'.join([f'pow(centroid_{i} - global_centroid_{i}, 2)' for i in range(num_features)])
        data_with_distances = data_with_centroids.withColumn(
            'distance_to_centroid',
            F.sqrt(F.expr(distance_calc))
        )
        
        # Step 10b: Calculate between-artist similarity per year
        yearly_stats = data_with_distances.groupBy('year').agg(
            F.avg('distance_to_centroid').alias('avg_distance'),
            F.stddev('distance_to_centroid').alias('std_distance'),
            F.expr('stddev(distance_to_centroid) / sqrt(count(*))').alias('se_distance'),
            F.count('*').alias('artist_count')
        )
    
    # Convert to Pandas DataFrame for visualization
    yearly_stats_pd = yearly_stats.toPandas().sort_values('year')
    
    return yearly_stats_pd


# Example usage
yearly_dispersion = calculate_yearly_dispersion(cluster_results, sample_size=0.9)
within_similarity = compute_artist_similarity(cluster_results, distance_type='within', sample_size=0.9)
between_similarity = compute_artist_similarity(cluster_results, distance_type='between', sample_size=0.9)

#%% visualize similarity trends
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def visualize_similarity_trends(data, distance_type='within'):
    """
    Visualize similarity trends with shaded error regions.

    Args:
        data: Pandas DataFrame containing yearly similarity metrics.
        distance_type: Type of similarity ('within' or 'between').
    """
    # Extract relevant columns
    metric = 'avg_distance' 
    std_metric = 'se_distance' 
    
    # Plot settings
    plt.figure(figsize=(12, 6))
    
    # Plot main line
    plt.plot(data['year'], data[metric], 'o-', markersize=5, 
             label=f'{distance_type.capitalize()}')
    
    # Add shaded error region
    lower_bound = np.maximum(0, data[metric] - data[std_metric])  # Ensure lower bound doesn't go below 0
    upper_bound = data[metric] + data[std_metric]
    plt.fill_between(data['year'], lower_bound, upper_bound, alpha=0.3)
    
    # Set y-axis limit to start at 0
    plt.ylim(bottom=0)
    
    # Add grid, labels, and legend
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Average Distance')
    plt.title(f'{distance_type.capitalize()} Innovation Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
visualize_similarity_trends(within_similarity, distance_type='within-artist')
visualize_similarity_trends(between_similarity, distance_type='between-artist')
visualize_similarity_trends(yearly_dispersion, distance_type='yearly')

# %%
# %% Optimized time series analysis with distributed computing
def distributed_time_series_analysis(cluster_data, sample_size=0.1, seed=None,visualize=True,
                                   bins=[0,2,100],
                                   forecast_years=0, cycle_years=40, holidays=['1966-01-01', '1993-01-01']):
    """
    Distributed version of time series analysis using PySpark for data processing
    """
    # Sample and prepare data using Spark
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Extract year and convert features to array
    prepared_data = sampled_data.withColumn(
        'year',
        F.year(F.to_timestamp('release_date'))
    ).withColumn(
        'features_array', 
        vector_to_array(F.col('features'))
    )
    
    # Get number of dimensions
    num_features = prepared_data.select(F.size('features_array').alias('num_features')).first()['num_features']
    
    # Extract features into columns
    for i in range(num_features):
        prepared_data = prepared_data.withColumn(f'feature_{i}', F.col('features_array')[i])
    
    # Calculate yearly centroids using Spark
    yearly_centroids = prepared_data.groupBy('year').agg(
        *[F.avg(F.col(f'feature_{i}')).alias(f'centroid_{i}') for i in range(num_features)]
    )
    # Join centroids and calculate distances
    data_with_centroids = prepared_data.join(yearly_centroids, on='year')
    distance_calc = '+'.join([f'pow(feature_{i} - centroid_{i}, 2)' for i in range(num_features)])
    data_with_distances = data_with_centroids.withColumn(
        'distance_to_centroid',
        F.sqrt(F.expr(distance_calc))
    )
    
    # Calculate bin distributions using raw distances
    bin_edges = bins
    bin_ranges = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    yearly_distributions = data_with_distances.groupBy('year').agg(
        *[
            F.sum(F.when((F.col('distance_to_centroid') >= bin_edges[i]) & 
                        (F.col('distance_to_centroid') < bin_edges[i+1]), 1)
                 .otherwise(0)).alias(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}")
            for i in range(len(bin_edges)-1)
        ]
    )
    # Convert to pandas for Prophet analysis
    yearly_distributions_pd = yearly_distributions.toPandas().sort_values('year')
    
    # Initialize Prophet results dictionary
    prophet_results = {}
    
    # Define holidays DataFrame
    holidays_df = None
    if holidays:
        holidays_df = pd.DataFrame({
            'holiday': 'innovation_peak',
            'ds': pd.to_datetime(holidays),
            'lower_window': -2,
            'upper_window': 2,
        })

    print(yearly_distributions_pd)
    
    # Fit Prophet models for each bin range
    for range_name in bin_ranges:
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(yearly_distributions_pd['year'].astype(str) + '-01-01'),
            'y': yearly_distributions_pd[range_name]
        })
        
        # Add streaming regressors
        df['year_since_streaming'] = (df['ds'].dt.year - 1999).clip(lower=0)
        df['streaming'] = (df['ds'] >= '1999-01-01').astype(int)
        
        # Initialize and fit Prophet model
        model = Prophet(
            mcmc_samples=200,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='additive',
            changepoint_prior_scale=0.001,
            holidays_prior_scale=10.0,
        )
        
        # Add custom seasonality and regressors
        model.add_seasonality(
            name='long_cycle',
            period=cycle_years * 365.25,
            fourier_order=3,
            prior_scale=1.0
        )
        model.add_regressor('streaming', prior_scale=10.0)
        model.add_regressor('year_since_streaming', prior_scale=10.0)
        
        # Fit model
        model.fit(df)
        
        # Make future predictions if requested
        future = model.make_future_dataframe(periods=forecast_years, freq='Y')
        future['year_since_streaming'] = (future['ds'].dt.year - 1999).clip(lower=0)
        future['streaming'] = (future['ds'] >= '1999-01-01').astype(int)
        forecast = model.predict(future)
        
        # Store results
        prophet_results[range_name] = {
            'model': model,
            'forecast': forecast,
            'actual': df
        }
        
        # Visualize results
        if visualize:
            model.plot_components(forecast)
            plt.suptitle(f'Analysis for Innovation Range {range_name}', fontsize=14)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 6))
            fig = model.plot(forecast)
            plt.title(f'Innovation Level Forecast for Range {range_name}\n(Including Regressor Effects)', fontsize=14)
            plt.tight_layout()
            plt.show()
    
    return prophet_results, data_with_distances,yearly_distributions_pd


# fitting and jointplot visualization
def visualize_fit_and_jointplot(prophet_results, bins=None, figsize=(10, 8)):
    """
    Visualize the fit of Prophet models and create joint plots of predicted vs observed data.

    Args:
        prophet_results: Dictionary containing Prophet models and results (output of time_series_analysis).
        bins: Optional list of bin ranges for filtering specific results. If None, use all bins.
        figsize: Tuple specifying the figure size for plots.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Get all available bins if not specified
    available_bins = list(prophet_results.keys())
    bins_to_plot = bins if bins is not None else available_bins

    # Iterate through each range and visualize
    for range_name in bins_to_plot:
        if range_name not in prophet_results:
            print(f"Warning: Range {range_name} not found in prophet_results")
            continue
            
        result = prophet_results[range_name]
        model = result['model']
        forecast = result['forecast']
        actual = result['actual']
        
        # Merge forecast and actual data for comparison
        merged = forecast[['ds', 'yhat']].merge(actual[['ds', 'y']], on='ds', how='inner')
        merged['residual'] = merged['y'] - merged['yhat']
        
        # Create jointplot for predicted vs observed values
        g = sns.jointplot(
            data=merged,
            x='yhat',
            y='y',
            kind='reg',
            height=8,
            marginal_kws=dict(bins=30, fill=True),
            joint_kws=dict(scatter_kws={'alpha': 0.5}),
        )
        
        # Add correlation and MAE to the plot
        r2 = np.corrcoef(merged['yhat'], merged['y'])[0, 1]**2
        mae = np.mean(np.abs(merged['residual']))
        g.ax_joint.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
                        transform=g.ax_joint.transAxes, verticalalignment='top')
        
        # Titles and labels
        g.fig.suptitle(f"Joint Plot for Range: {range_name}", y=1.02)
        g.set_axis_labels("Predicted (yhat)", "Observed (y)")
        plt.show()

        # Plot residuals
        plt.figure(figsize=figsize)
        sns.histplot(merged['residual'], kde=True, bins=30, color='#3C76AF')
        plt.title(f"Residual Distribution for Range: {range_name}", fontsize=14)
        plt.xlabel("Residuals (Observed - Predicted)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Plot time series fit
        plt.figure(figsize=figsize)
        plt.scatter(actual['ds'], actual['y'], label='Observed', color='black',alpha=0.8, s=20)
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='#3C76AF',linewidth=2)
        plt.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color='#3a76AF', alpha=0.2, label='Prediction Interval'
        )
        plt.title(f"Time Series Fit for Range: {range_name}", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"Range: {range_name}")

def extract_and_visualize_regressor_coefficients_with_ci(prophet_results, regressors, bins=None):
    """
    Extract and visualize regressor coefficients with credible intervals (CI) from Prophet models.
    
    Args:
        prophet_results: Dictionary containing Prophet models and results.
        regressors: List of regressors to analyze.
        bins: Optional list of range names to include in the analysis. If None, use all bins.
    """
    import numpy as np
    from prophet.utilities import regressor_coefficients
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Get all available bins if not specified
    available_bins = list(prophet_results.keys())
    bins_to_analyze = bins if bins is not None else available_bins

    coefficients = []
    

    for range_name in bins_to_analyze:
        if range_name not in prophet_results:
            print(f"Warning: Range {range_name} not found in prophet_results")
            continue
            
        result = prophet_results[range_name]
        model = result['model']
        regressor_data = regressor_coefficients(model)
        
        if 'coef' in regressor_data.columns:
            for _, row in regressor_data.iterrows():
                if row['regressor'] in regressors:
                    coefficients.append({
                        'range': range_name,
                        'regressor': 'intercept' if row['regressor'] == 'streaming' else 'slope',
                        'coefficient': row['coef'],
                        'lower_bound': row['coef_lower'] if 'coef_lower' in row else np.nan,
                        'upper_bound': row['coef_upper'] if 'coef_upper' in row else np.nan
                    })

    coefficients_df = pd.DataFrame(coefficients)
    print("\nRegressor Coefficients Summary:")
    print(coefficients_df)
    
    # Create visualization
    plt.figure(figsize=(50, 6))
    
    # Generate colors dynamically based on number of ranges
    unique_ranges = coefficients_df['range'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_ranges)))
    range_colors = dict(zip(unique_ranges, colors))

    # Plot coefficients with error bars
    for i, regressor in enumerate(['intercept', 'slope']):
        regressor_data = coefficients_df[coefficients_df['regressor'] == regressor]
        x_positions = np.arange(len(regressor_data)) + i * 0.45
        
        for j, (idx, row) in enumerate(regressor_data.iterrows()):
            plt.errorbar(x_positions[j], 
                        row['coefficient'],
                        yerr=[[row['coefficient'] - row['lower_bound']],
                              [row['upper_bound'] - row['coefficient']]],
                        fmt='o',
                        capsize=5,
                        capthick=2,
                        label=f"{regressor} ({row['range']})",
                        markersize=8,
                        color=range_colors[row['range']])

    # Customize plot
    plt.xticks(np.arange(len(unique_ranges)) + 0.175, unique_ranges)
    plt.xlabel('Innovation Level')
    plt.ylabel('Coefficient Value')
    plt.title('Regressor Coefficients with 95% CI')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
def extract_and_visualize_regressor_coefficients_with_ci_numeric(prophet_results, regressors, bins=None):
    """
    Extract and visualize regressor coefficients with credible intervals (CI) from Prophet models.
    
    Args:
        prophet_results: Dictionary containing Prophet models and results.
        regressors: List of regressors to analyze.
        bins: Optional list of range names to include in the analysis. If None, use all bins.
    """
    import numpy as np
    from prophet.utilities import regressor_coefficients
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Get all available bins if not specified
    available_bins = list(prophet_results.keys())
    bins_to_analyze = bins if bins is not None else available_bins

    coefficients = []

    for range_name in bins_to_analyze:
        if range_name not in prophet_results:
            print(f"Warning: Range {range_name} not found in prophet_results")
            continue
            
        result = prophet_results[range_name]
        model = result['model']
        regressor_data = regressor_coefficients(model)
        
        if 'coef' in regressor_data.columns:
            for _, row in regressor_data.iterrows():
                if row['regressor'] in regressors:
                    coefficients.append({
                        'range': float(range_name.split('-')[0]),  # Convert range to numeric
                        'range_original': range_name,  # Keep original range name
                        'regressor': 'intercept' if row['regressor'] == 'streaming' else 'slope',
                        'coefficient': row['coef'],
                        'lower_bound': row['coef_lower'] if 'coef_lower' in row else np.nan,
                        'upper_bound': row['coef_upper'] if 'coef_upper' in row else np.nan
                    })

    coefficients_df = pd.DataFrame(coefficients)
    print("\nRegressor Coefficients Summary:")
    print(coefficients_df)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot coefficients with error bands for each regressor type
    for regressor in ['intercept', 'slope']:
        regressor_data = coefficients_df[coefficients_df['regressor'] == regressor].sort_values('range')
        
        plt.plot(regressor_data['range'], 
                regressor_data['coefficient'], 
                '-o',
                label=regressor)
        
        plt.fill_between(regressor_data['range'],
                        regressor_data['lower_bound'],
                        regressor_data['upper_bound'],
                        alpha=0.2,
                        label=f'95% CI ({regressor})')

    
    # Customize plot
    plt.xlabel('Innovation Level')
    plt.ylabel('Coefficient Value')
    plt.title('Impact of Streaming on Innovation')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Set x-ticks to show all original range labels
    sorted_ranges = sorted(coefficients_df['range'].unique())
    range_mapping = coefficients_df.sort_values('range').drop_duplicates('range')[['range', 'range_original']]
    plt.xticks(sorted_ranges, range_mapping['range_original'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


#visualize_fit_and_jointplot(prophet_results)
extract_and_visualize_regressor_coefficients_with_ci_numeric(
    prophet_results,
    regressors=['streaming', 'year_since_streaming']
)


# %%

# Example usage
prophet_results, processed_data,yearly_distributions_pd = distributed_time_series_analysis(
    cluster_results, 
    sample_size=0.99,
    bins=[i/10 for i in range(0,60,5)],
    forecast_years=0,
    cycle_years=40,
    holidays=['1966-01-01', '1993-01-01'],
    visualize=True
)

# %%
def visualize_innovation_levels_over_time(yearly_distributions_pd, figsize=(12, 6)):
    """
    Visualize the distribution of innovation levels over time.
    
    Args:
        yearly_distributions_pd: Pandas DataFrame containing yearly distributions of innovation levels
        figsize: Tuple specifying figure size
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get innovation level columns (excluding 'year')
    level_columns = [col for col in yearly_distributions_pd.columns if col != 'year']
    
    # Create colormap
    n_levels = len(level_columns)
    colors = plt.cm.viridis(np.linspace(0, 1, n_levels))
    
    # Plot each innovation level
    for level, color in zip(level_columns, colors):
        plt.plot(yearly_distributions_pd['year'], 
                yearly_distributions_pd[level],
                '-o', markersize=4, color=color,
                label=f'Level {level}')
    
    # Customize plot
    plt.xlabel('Year')
    plt.ylabel('Number of Songs')
    plt.title('Distribution of Innovation Levels Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Example usage
# Merge bins into new ranges [0-1.5, 1.5-100]
import copy
yearly_distributions_pd2 = copy.deepcopy(yearly_distributions_pd)
new_ranges = ['0.0-1.5', '1.5-100.0']

# Initialize new columns with zeros
yearly_distributions_pd2['0.0-1.5'] = 0
yearly_distributions_pd2['1.5-100.0'] = 0

# Iterate through original columns and add to appropriate new range
for col in yearly_distributions_pd2.columns:
    if col != 'year':
        # Extract range bounds from column name
        lower, upper = map(float, col.split('-'))
        
        # Add to appropriate new range
        if upper <= 1.5:
            yearly_distributions_pd2['0.0-1.5'] += yearly_distributions_pd2[col]
        else:
            yearly_distributions_pd2['1.5-100.0'] += yearly_distributions_pd2[col]

# Keep only year and new range columns            
yearly_distributions_pd2 = yearly_distributions_pd2[['year'] + new_ranges]

visualize_innovation_levels_over_time(yearly_distributions_pd2)

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.ml.functions import vector_to_array
def plot_yearly_distribution_animation_distributed(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Create an animated plot showing the evolution of song distribution in PCA space over time,
    using distributed computing for both data preparation and bin counting.
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata.
        sample_size: Fraction of data to sample.
        grid_size: Number of grid cells in each dimension.
        seed: Random seed for sampling.
    """

    try:
        # Sample and prepare data using Spark
        sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
        
        # Extract year and convert features to array
        prepared_data = sampled_data.withColumn(
            'year',
            F.year(F.to_timestamp('release_date'))
        ).withColumn(
            'features_array',
            vector_to_array(F.col('features'))
        )
        
        # Get number of dimensions and extract features into columns
        num_features = prepared_data.select(F.size('features_array').alias('num_features')).first()['num_features']
        for i in range(num_features):
            prepared_data = prepared_data.withColumn(f'feature_{i}', F.col('features_array')[i])
        
        # Calculate min/max for grid edges using Spark
        feature_stats = prepared_data.agg(
            F.min('feature_0').alias('x_min'),
            F.max('feature_0').alias('x_max'),
            F.min('feature_1').alias('y_min'),
            F.max('feature_1').alias('y_max')
        ).collect()[0]
        
        x_edges = np.linspace(feature_stats['x_min'], feature_stats['x_max'], grid_size)
        y_edges = np.linspace(feature_stats['y_min'], feature_stats['y_max'], grid_size)
        
        # Broadcast edges to workers
        x_edges_bc = cluster_data.sql_ctx.sparkSession.sparkContext.broadcast(x_edges)
        y_edges_bc = cluster_data.sql_ctx.sparkSession.sparkContext.broadcast(y_edges)
        
        # Assign bin indices to each song
        def assign_bin(x, edges):
            return np.digitize(x, edges) - 1  # Convert bin index to zero-based
        
        assign_bin_udf = udf(lambda arr: (assign_bin(float(arr[0]), x_edges_bc.value), assign_bin(float(arr[1]), y_edges_bc.value)), ArrayType(IntegerType()))
        prepared_data = prepared_data.withColumn('bin_indices', assign_bin_udf(array('feature_0', 'feature_1')))
        
        # Explode bin indices into separate columns
        prepared_data = prepared_data.withColumn('x_bin', F.col('bin_indices')[0]).withColumn('y_bin', F.col('bin_indices')[1])
        
        # Calculate counts per bin per year
        bin_counts = prepared_data.groupBy('year', 'x_bin', 'y_bin').count().cache()
        
        # Convert to Pandas for visualization
        bin_counts_pd = bin_counts.toPandas()

        # Get unique years
        years = sorted(bin_counts_pd['year'].unique())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        def update(frame):
            ax.clear()
            year = years[frame]
            year_data = bin_counts_pd[bin_counts_pd['year'] == year]
            
            # Create grid for heatmap
            heatmap = np.zeros((grid_size, grid_size))
            for _, row in year_data.iterrows():
                x_bin, y_bin, count = row['x_bin'], row['y_bin'], row['count']
                if 0 <= x_bin < grid_size and 0 <= y_bin < grid_size:
                    heatmap[x_bin, y_bin] = count
            
            # Plot heatmap
            im = ax.imshow(
                heatmap,
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
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(years),
            interval=100,
            repeat=True
        )
        
        plt.suptitle('Evolution of Music Distribution in PCA Space', y=1.02, fontsize=16)
        plt.tight_layout()
        
        # Save animation
        anim.save('music_distribution_evolution_distributed.gif', writer='pillow')
        plt.show()

        # Clean up
        bin_counts.unpersist()
        
    except Exception as e:
        print(f"Error in plot_yearly_distribution_animation_distributed: {e}")
        raise

# Example usage 
plot_yearly_distribution_animation_distributed(cluster_results, sample_size=0.01)
# %% animation all; PCA space with distributed computing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.ml.functions import vector_to_array

def plot_yearly_distribution_animation(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Create an animated plot showing the evolution of song distribution in PCA space over time,
    using distributed computing for data processing
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata
        sample_size: Fraction of data to sample
        grid_size: Number of grid cells in each dimension
        seed: Random seed for sampling
    """
    # Sample and prepare data using Spark
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Extract year and convert features to array
    prepared_data = sampled_data.withColumn(
        'year',
        F.year(F.to_timestamp('release_date'))
    ).withColumn(
        'features_array',
        vector_to_array(F.col('features'))
    )
    
    # Get number of dimensions and extract features into columns
    num_features = prepared_data.select(F.size('features_array').alias('num_features')).first()['num_features']
    for i in range(num_features):
        prepared_data = prepared_data.withColumn(f'feature_{i}', F.col('features_array')[i])
    
    # Calculate min/max for grid edges using Spark
    feature_stats = prepared_data.agg(
        F.min('feature_0').alias('x_min'),
        F.max('feature_0').alias('x_max'),
        F.min('feature_1').alias('y_min'),
        F.max('feature_1').alias('y_max')
    ).collect()[0]
    
    x_edges = np.linspace(feature_stats['x_min'], feature_stats['x_max'], grid_size)
    y_edges = np.linspace(feature_stats['y_min'], feature_stats['y_max'], grid_size)
    
    # Convert to pandas for visualization
    df = prepared_data.toPandas()
    pca_coords = df[['feature_0', 'feature_1']].values
    years = df['year'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
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
        interval=100,
        repeat=True
    )
    
    plt.suptitle('Evolution of Music Distribution in PCA Space', y=1.02, fontsize=16)
    
    # Save animation
    anim.save('music_distribution_evolution.gif', writer='pillow')
    plt.show()

# Example usage 
plot_yearly_distribution_animation(cluster_results, sample_size=0.01)
# %%

def plot_yearly_distribution_animation_distributed(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Create an animated plot showing the evolution of song distribution in PCA space over time,
    using distributed computing for both data preparation and bin counting.
    
    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata.
        sample_size: Fraction of data to sample.
        grid_size: Number of grid cells in each dimension.
        seed: Random seed for sampling.
    """
    from pyspark.sql import Window
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import animation
    from pyspark.sql.functions import col

    # Sample and prepare data using Spark
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Extract year and convert features to array
    prepared_data = sampled_data.withColumn(
        'year',
        F.year(F.to_timestamp('release_date'))
    ).withColumn(
        'features_array',
        vector_to_array(F.col('features'))
    )
    
    # Get number of dimensions and extract features into columns
    num_features = prepared_data.select(F.size('features_array').alias('num_features')).first()['num_features']
    for i in range(num_features):
        prepared_data = prepared_data.withColumn(f'feature_{i}', F.col('features_array')[i])
    
    # Calculate min/max for grid edges using Spark
    feature_stats = prepared_data.agg(
        F.min('feature_0').alias('x_min'),
        F.max('feature_0').alias('x_max'),
        F.min('feature_1').alias('y_min'),
        F.max('feature_1').alias('y_max')
    ).collect()[0]
    
    x_edges = np.linspace(feature_stats['x_min'], feature_stats['x_max'], grid_size)
    y_edges = np.linspace(feature_stats['y_min'], feature_stats['y_max'], grid_size)
    
    # Broadcast edges to workers
    x_edges_bc = cluster_data.sql_ctx.sparkSession.sparkContext.broadcast(x_edges)
    y_edges_bc = cluster_data.sql_ctx.sparkSession.sparkContext.broadcast(y_edges)
    
    # Assign bin indices to each song
    def assign_bin(x, edges):
        return np.digitize(x, edges) - 1  # Convert bin index to zero-based
    
    assign_bin_udf = F.udf(lambda arr: (assign_bin(arr[0], x_edges_bc.value), assign_bin(arr[1], y_edges_bc.value)), ArrayType(IntegerType()))
    prepared_data = prepared_data.withColumn('bin_indices', assign_bin_udf(F.array('feature_0', 'feature_1')))
    
    # Explode bin indices into separate columns
    prepared_data = prepared_data.withColumn('x_bin', F.col('bin_indices')[0]).withColumn('y_bin', F.col('bin_indices')[1])
    
    # Calculate counts per bin per year
    bin_counts = prepared_data.groupBy('year', 'x_bin', 'y_bin').count()
    
    # Convert to Pandas for visualization
    bin_counts_pd = bin_counts.toPandas()

    # Get unique years
    years = sorted(bin_counts_pd['year'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    def update(frame):
        ax.clear()
        year = years[frame]
        year_data = bin_counts_pd[bin_counts_pd['year'] == year]
        
        # Create grid for heatmap
        heatmap = np.zeros((grid_size, grid_size))
        for _, row in year_data.iterrows():
            x_bin, y_bin, count = row['x_bin'], row['y_bin'], row['count']
            if 0 <= x_bin < grid_size and 0 <= y_bin < grid_size:
                heatmap[y_bin, x_bin] = count
        
        # Plot heatmap
        im = ax.imshow(
            heatmap.T,
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
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(years),
        interval=100,
        repeat=True
    )
    
    plt.suptitle('Evolution of Music Distribution in PCA Space', y=1.02, fontsize=16)
    
    # Save animation
    anim.save('music_distribution_evolution_distributed.gif', writer='pillow')
    plt.show()

# Example usage 
plot_yearly_distribution_animation_distributed(cluster_results, sample_size=0.01)
# %%
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_yearly_distribution_animation_distributed(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Create an animated plot showing the evolution of song distribution in PCA space over time,
    using Spark distributed computing for both data preparation and bin counting.

    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata.
        sample_size: Fraction of data to sample.
        grid_size: Number of grid cells in each dimension.
        seed: Random seed for sampling.

    Returns:
        Spark DataFrame containing yearly grid counts, to be used for further analysis or visualization.
    """
    # Step 1: Sample data
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Step 2: Extract year and prepare PCA features
    prepared_data = sampled_data.withColumn(
        'year', F.year(F.to_timestamp('release_date'))
    ).withColumn(
        'features_array', vector_to_array(F.col('features'))
    )
    
    # Step 3: Determine grid edges (min/max)
    feature_stats = prepared_data.agg(
        F.min('features_array')[0].alias('x_min'),
        F.max('features_array')[0].alias('x_max'),
        F.min('features_array')[1].alias('y_min'),
        F.max('features_array')[1].alias('y_max')
    ).collect()[0]
    
    x_edges = np.linspace(feature_stats['x_min'], feature_stats['x_max'], grid_size + 1)
    y_edges = np.linspace(feature_stats['y_min'], feature_stats['y_max'], grid_size + 1)

    # Step 4: Assign bin indices using Spark
    def assign_bin(x, edges):
        return int(np.digitize(x, edges) - 1)  # Convert bin index to zero-based

    assign_bin_udf = F.udf(lambda arr: (assign_bin(arr[0], x_edges), assign_bin(arr[1], y_edges)), ArrayType(IntegerType()))
    prepared_data = prepared_data.withColumn('bin_indices', assign_bin_udf(F.col('features_array')))
    prepared_data = prepared_data.withColumn('x_bin', F.col('bin_indices')[0]).withColumn('y_bin', F.col('bin_indices')[1])

    # Step 5: Count occurrences per bin per year
    bin_counts = prepared_data.groupBy('year', 'x_bin', 'y_bin').agg(
        F.count('*').alias('count')
    )

    # Collect unique years
    years = bin_counts.select('year').distinct().orderBy('year').collect()
    unique_years = [row['year'] for row in years]

    # Step 6: Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    # Pre-fetch data for all years using Spark (minimizing collect calls)
    bin_counts_dict = {year: bin_counts.filter(F.col('year') == year).collect() for year in unique_years}

    def update(frame):
        ax.clear()
        year = unique_years[frame]
        year_data = bin_counts_dict[year]

        # Create a grid for the heatmap
        heatmap = np.zeros((grid_size, grid_size))
        for row in year_data:
            x_bin, y_bin, count = row['x_bin'], row['y_bin'], row['count']
            if 0 <= x_bin < grid_size and 0 <= y_bin < grid_size:
                heatmap[y_bin, x_bin] = count

        # Plot the heatmap
        im = ax.imshow(
            heatmap.T,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin='lower',
            cmap='viridis',
            aspect='auto'
        )
        ax.set_title(f'Year: {year}')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')

        # Add colorbar on the first frame
        if frame == 0:
            fig.colorbar(im, cax=cbar_ax, label='Number of Songs')

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(unique_years),
        interval=200,
        repeat=True
    )

    plt.suptitle('Evolution of Music Distribution in PCA Space', y=1.02, fontsize=16)

    # Save the animation
    anim.save('music_distribution_evolution_distributed.gif', writer='pillow')
    plt.show()

    return bin_counts  # Return Spark DataFrame containing yearly bin counts

# Example usage
result = plot_yearly_distribution_animation_distributed(cluster_results, sample_size=0.01, grid_size=50)
# %% heatmap evolution
GRID_SIZE = 70
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def calculate_yearly_bin_counts(cluster_data, sample_size=0.1, grid_size=50, seed=42):
    """
    Calculate yearly bin counts for PCA features using Spark.

    Args:
        cluster_data: Spark DataFrame containing PCA features and metadata.
        sample_size: Fraction of data to sample.
        grid_size: Number of grid cells in each dimension.
        seed: Random seed for sampling.

    Returns:
        Spark DataFrame with year, x_bin, y_bin, and count for each bin.
    """
    # Step 1: Sample data
    sampled_data = cluster_data.sample(withReplacement=False, fraction=sample_size, seed=seed)

    # Step 2: Extract year and prepare PCA features
    prepared_data = sampled_data.withColumn(
        'year', F.year(F.to_timestamp('release_date'))
    ).withColumn(
        'features_array', vector_to_array(F.col('features'))
    )

    # Step 3: Determine feature ranges
    feature_stats = prepared_data.agg(
        F.min('features_array')[0].alias('x_min'),
        F.max('features_array')[0].alias('x_max'),
        F.min('features_array')[1].alias('y_min'),
        F.max('features_array')[1].alias('y_max')
    ).collect()[0]
    
    x_min, x_max = feature_stats['x_min'], feature_stats['x_max']
    y_min, y_max = feature_stats['y_min'], feature_stats['y_max']

    # Step 4: Define UDF for interval assignment
    def assign_bin(value, min_val, max_val, grid_size):
        """Map a value to a bin index based on the grid range."""
        #return int((value - min_val) / (max_val - min_val) * grid_size)
        if max_val == min_val:  # Handle edge case where range is zero
            return 0
        normalized_value = (value - min_val) / (max_val - min_val)  # Normalize value to [0, 1]
        bin_index = int(normalized_value * grid_size)  # Scale to [0, grid_size]
        return min(max(bin_index, 0), grid_size - 1)  # Clamp to valid range

    assign_bin_udf_x = F.udf(lambda x: assign_bin(x, x_min, x_max, grid_size), IntegerType())
    assign_bin_udf_y = F.udf(lambda y: assign_bin(y, y_min, y_max, grid_size), IntegerType())

    # Step 5: Assign bins
    prepared_data = prepared_data.withColumn('x_bin', assign_bin_udf_x(F.col('features_array')[0]))
    prepared_data = prepared_data.withColumn('y_bin', assign_bin_udf_y(F.col('features_array')[1]))

    # Step 6: Group by year and bins, then count
    bin_counts = prepared_data.groupBy('year', 'x_bin', 'y_bin').agg(
        F.count('*').alias('count')
    )

    return bin_counts

# Calculate bin counts using Spark
bin_counts = calculate_yearly_bin_counts(cluster_results, sample_size=0.99, grid_size=GRID_SIZE)
bin_counts.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_yearly_heatmaps_animation(bin_counts, grid_size, filename="heatmap_animation.gif", year_list=None):
    """
    Create an animated heatmap showing the normalized distribution of counts across x_bin and y_bin over years.
    
    Args:
        bin_counts: Spark DataFrame
            DataFrame containing columns ['year', 'x_bin', 'y_bin', 'count'].
        grid_size: int
            Size of the grid for both x and y dimensions.
        filename: str
            Output filename for the animated GIF.
        year_list: list, optional
            List of years to include in the animation. If None, include all years.
    """
    bin_counts_pd = bin_counts.toPandas()
    x_bins = range(grid_size)  # Define x_bin range
    y_bins = range(grid_size)  # Define y_bin range
    
    if year_list is None:
        years = sorted(bin_counts_pd['year'].unique())  # Get all years
    else:
        years = sorted(year_list)  # Use provided years
    
    # Standardize counts within each year
    bin_counts_pd['normalized_count'] = bin_counts_pd.groupby('year')['count'].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    
    # Prepare figure for animation
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap_data = np.zeros((grid_size, grid_size))  # Initialize an empty matrix
    
    # Configure heatmap style
    sns_heatmap = sns.heatmap(
        heatmap_data, 
        cmap='YlOrRd', 
        cbar_kws={'label': 'Normalized Count'}, 
        annot=True, 
        fmt='.2f', 
        mask=(heatmap_data == 0),  # Mask zero values
        vmin=0,
        vmax=1,  # Normalized range
        ax=ax
    )
    
    # Set axis labels and ticks with fewer tick marks
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    
    # Show fewer ticks (e.g. only 5 ticks)
    tick_positions = np.linspace(0, grid_size, 5)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    # Set tick labels as integers
    tick_labels = [int(x) for x in tick_positions]
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    def update(year):
        """Update the heatmap for a given year."""
        # Clear the heatmap data
        heatmap_data.fill(0)
        
        # Filter data for current year
        year_data = bin_counts_pd[bin_counts_pd['year'] == year]
        
        # Fill normalized count data
        for _, row in year_data.iterrows():
            x, y, norm_count = int(row['x_bin']), int(row['y_bin']), row['normalized_count']
            if 0 <= x < grid_size and 0 <= y < grid_size:
                heatmap_data[y, x] = norm_count  # Fill corresponding position
        
        # Update heatmap data
        sns_heatmap.collections[0].set_array(heatmap_data.flatten())
        
        # Update the title dynamically
        ax.set_title(f'Heatmap for Year {year}')
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=years, repeat=False, interval=20)
    
    # Save as GIF
    gif_writer = PillowWriter()  # Slower FPS for better visualization
    gif_filename = filename
    ani.save(gif_filename, writer=gif_writer)
    print(f"Animation saved as {gif_filename}")


plot_yearly_heatmaps_animation(bin_counts, grid_size=GRID_SIZE, filename="heatmap_animation.gif")

# %% sample

def plot_pca_sample_by_year(cluster_data, start_year=None, end_year=None, sample_size=0.1, seed=42):
    """
    Create a scatter plot of sampled songs in PCA space, colored by cluster, for a specified year range
    
    Args:
        cluster_data: PySpark DataFrame with features and metadata
        start_year: Start year for filtering (inclusive). If None, no lower bound
        end_year: End year for filtering (inclusive). If None, no upper bound
        sample_size: Fraction of songs to sample (default 0.1)
        seed: Random seed for sampling
    """
    # Filter data by year range
    filtered_data = cluster_data
    if start_year is not None:
        filtered_data = filtered_data.filter(F.year(F.to_timestamp('release_date')) >= start_year)
    if end_year is not None:
        filtered_data = filtered_data.filter(F.year(F.to_timestamp('release_date')) <= end_year)
    
    # Sample data
    sampled_data = filtered_data.sample(withReplacement=False, fraction=sample_size, seed=seed)
    
    # Convert to pandas for visualization
    df = sampled_data.select('features', 'prediction').toPandas()
    
    # Extract PCA coordinates
    pca_coords = np.vstack(df['features'].values)
    
    # Define markers and colors
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '8', '*', 'H']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
              
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster with different marker and color
    for cluster_id in df['prediction'].unique():
        mask = df['prediction'] == cluster_id
        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                   marker=markers[cluster_id % len(markers)],
                   c=[colors[cluster_id % len(colors)]],
                   label=f'Cluster {cluster_id}',
                   alpha=0.6, s=50)
    
    # Add labels and title
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # Create title based on year range
    if start_year is not None and end_year is not None:
        year_range = f"{start_year}-{end_year}"
    elif start_year is not None:
        year_range = f"{start_year}+"
    elif end_year is not None:
        year_range = f"Until {end_year}"
    else:
        year_range = "All Years"
    
    plt.title(f'PCA Distribution of Songs ({year_range}) by Cluster (Sample Size: {sample_size*100:.1f}%)')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
plot_pca_sample_by_year(cluster_results, start_year=1900, end_year=2020, sample_size=0.01)



# %%
def plot_pca_sample(features_pca, sample_size=0.1, seed=None):
    """
    Plot sampled PCA points in 2D space
    
    Args:
        features_pca: DataFrame with PCA features
        sample_size: Fraction of data to sample (default 0.1)
        seed: Random seed for sampling (optional)
    """
    # Sample data
    if seed is None:
        sampled_data = features_pca.sample(False, sample_size)
    else:
        sampled_data = features_pca.sample(False, sample_size, seed=seed)
    
    # Convert to pandas and extract coordinates
    df = sampled_data.toPandas()
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

# Example usage
plot_pca_sample(features_pca, sample_size=0.01)

# %%
def plot_pca_sample(features_pca, df, sample_size=0.1, seed=None):
    """
    Plot sampled PCA points in 2D space, colored by year and label points with high second component
    Args:
        features_pca: DataFrame with PCA features
        df: Original dataframe with release dates
        sample_size: Fraction of data to sample (default 0.1)
        seed: Random seed for sampling (optional)
    """
    # Add tmp_id for joining
    features_pca_with_id = features_pca.withColumn("tmp_id", F.monotonically_increasing_id())
    df_with_id = df.withColumn("tmp_id", F.monotonically_increasing_id())
    
    # Join with original df to get release dates and names
    joined_data = features_pca_with_id.join(df_with_id.select("tmp_id", "release_date", "name"), on="tmp_id")
    
    # Convert to pandas and extract coordinates
    df_pd = joined_data.toPandas()
    coords = np.vstack(df_pd['features'].values)
    
    # Handle invalid date formats
    years = pd.to_datetime(df_pd['release_date'], errors='coerce').dt.year
    # Drop rows with invalid years
    valid_mask = ~years.isna()
    coords = coords[valid_mask]
    years = years[valid_mask]
    names = df_pd['name'][valid_mask]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=years, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Year')
    
    # Add labels for points with second component > 30
    for i, (x, y) in enumerate(coords):
        if y > 30:
            plt.annotate(names.iloc[i], (x, y), xytext=(5, 5), textcoords='offset points')
            print(f"Song: {names.iloc[i]}, Year: {years.iloc[i]}, PC1: {x:.2f}, PC2: {y:.2f}")
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Distribution of Songs by Year')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
plot_pca_sample(features_pca, df)
# %%