
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
from prophet import Prophet


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
# 1. PCA: find optimal number of components
optimal_n, features_pca, explained_variances, cumulative_variance, model_pca = find_optimal_pca_components(features,k=2)
components_df = analyze_pca_composition(model_pca, feature_cols)
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
                np.sum((current_points - current_centroid) ** 2, axis=1))
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

# 使用历史距离
results_df = plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.1,bins=[0,1,1.5,2,2.5,100])
plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.1,bins=[0,2,100])

# 用年距离
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.3,bins=[0,1,1.5,2,2.5,100])
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.3,bins=[0,2,100])






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




#%% intercept | good
def analyze_innovation_seasonality(cluster_data, artist_name=None, sample_size=0.1, seed=None,
                                 bins=[0, 1, 2, 3, 4], distance_type='historical',
                                 forecast_years=10, cycle_years=40, holidays=None, changepoints=None):
    """
    Analyze innovation levels using Prophet with holidays and changepoints
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
        bins: Innovation level ranges
        distance_type: Type of distance to calculate ('historical' or 'internal')
        forecast_years: Number of years to forecast
        cycle_years: Length of cycle in years
        changepoints: List of dates for changepoints
    """
    from prophet import Prophet
    
    # Get innovation data using existing function
    results_df = plot_innovation_levels_over_time(cluster_data, artist_name, sample_size, 
                                                seed, bins, distance_type)
    

    # Define holidays
    holidays = pd.DataFrame({
        'holiday': 'innovation_peak',
        'ds': pd.to_datetime(holidays),
        'lower_window': -2,
        'upper_window': 2,
    })
    
    prophet_results = {}
    bin_ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    for range_name in bin_ranges:
        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(results_df.index.astype(str) + '-01-01'),
            'y': results_df[range_name]
        })
        
        # Add streaming regressors with time component
        df['year_since_1999'] = (df['ds'].dt.year - 1999) * (df['ds'] >= '1999-01-01')
        df['year_since_2015'] = (df['ds'].dt.year - 2015) * (df['ds'] >= '2015-01-01')
        df['streaming1'] = (df['ds'] >= '1999-01-01').astype(int)
        df['streaming2'] = (df['ds'] >= '2015-01-01').astype(int)

        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            holidays_prior_scale=10.0,
            changepoints=changepoints
        )
        
        # Add custom seasonality for long-term cycles
        model.add_seasonality(
            name='long_cycle',
            period=cycle_years * 365.25,
            fourier_order=3
        )
        
        # Add streaming regressors with both intercept and slope
        model.add_regressor('streaming1')
        model.add_regressor('streaming2')
        model.add_regressor('year_since_streaming1')
        model.add_regressor('year_since_streaming2')
        
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(
            periods=forecast_years,
            freq='Y',
            include_history=True
        )
        # Add streaming regressors to future dataframe
        future['streaming1'] = (future['ds'] >= '1999-01-01').astype(int)
        future['streaming2'] = (future['ds'] >= '2015-01-01').astype(int)
        future['year_since_streaming1'] = (future['ds'].dt.year - 1999) * (future['ds'] >= '1999-01-01')
        future['year_since_streaming2'] = (future['ds'].dt.year - 2015) * (future['ds'] >= '2015-01-01')
        
        forecast = model.predict(future)
        
        # Store results
        prophet_results[range_name] = {
            'model': model,
            'forecast': forecast,
            'actual': df
        }
        # Plot components
        model.plot_components(forecast)
        plt.suptitle(f'Analysis for Innovation Range {range_name}')
        plt.tight_layout()
        plt.show()
        
        # Plot forecast with changepoints
        plt.figure(figsize=(12, 6))
        fig = model.plot(forecast)
        
        # Add vertical lines for changepoints
        ax = fig.gca()
        for cp in model.changepoints:
            ax.axvline(x=cp, color='r', alpha=0.2, linestyle='--')

        from prophet.plot import add_changepoints_to_plot
        a = add_changepoints_to_plot(ax, model, forecast)

        plt.title(f'Innovation Level Forecast for Range {range_name}\nRed dashed lines indicate trend changepoints')
        plt.tight_layout()
        plt.show()
        
        # Plot changepoint effects
        deltas = model.params['delta'].mean(0)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.bar(range(len(deltas)), deltas)
        ax.set_title('Magnitude of Changepoint Effects')
        ax.set_xlabel('Changepoint Number')
        ax.set_ylabel('Effect Size')
        plt.tight_layout()
        plt.show()
    
    return prophet_results

# Example usage:
prophet_results = analyze_innovation_seasonality(
    cluster_results, 
    distance_type='internal',
    sample_size=0.1,
    bins=[0,1,1.5,2,2.5,100],
    forecast_years=10,
    cycle_years=45,
    seed=77,
    holidays=['1966-01-01', '1993-01-01'],
    changepoints=[f'{year}-01-01' for year in range(1995, 2021)]
)

# %%
def analyze_innovation_seasonality(cluster_data, artist_name=None, sample_size=0.1, seed=None,
                                   bins=[0, 1, 2, 3, 4], distance_type='historical',
                                   forecast_years=10, cycle_years=40, holidays=None):
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
        df['year_since_1999'] = (df['ds'].dt.year - 1999).clip(lower=0)
        df['year_since_2015'] = (df['ds'].dt.year - 2015).clip(lower=0)
        df['streaming1'] = (df['ds'] >= '1999-01-01').astype(int)
        df['streaming2'] = (df['ds'] >= '2015-01-01').astype(int)
        
        # Step 4: Initialize Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            holidays_prior_scale=10.0,
        )
        
        # Add custom seasonality
        model.add_seasonality(
            name='long_cycle',
            period=cycle_years * 365.25,
            fourier_order=3
        )
        
        # Add regressors for intercept and slope changes
        model.add_regressor('streaming1')
        model.add_regressor('streaming2')
        model.add_regressor('year_since_1999')
        model.add_regressor('year_since_2015')
        
        # Step 5: Fit the model
        model.fit(df)
        
        # Step 6: Make future predictions
        future = model.make_future_dataframe(periods=forecast_years, freq='Y')
        future['year_since_1999'] = (future['ds'].dt.year - 1999).clip(lower=0)
        future['year_since_2015'] = (future['ds'].dt.year - 2015).clip(lower=0)
        future['streaming1'] = (future['ds'] >= '1999-01-01').astype(int)
        future['streaming2'] = (future['ds'] >= '2015-01-01').astype(int)
        
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
        
        # Plot regressor effects
        regressors = ['streaming1', 'streaming2', 'year_since_1999', 'year_since_2015']
        regressor_effects = []
        
        # Safely access regressor coefficients from model parameters
        for name in regressors:
            try:
                # Convert params['beta'] to a dictionary if it's a Series/DataFrame
                beta_params = dict(zip(model.extra_regressors.keys(), 
                                     model.params['beta'].flatten()))
                effect = beta_params.get(name, 0)
                regressor_effects.append(effect)
            except Exception as e:
                print(f"Warning: Could not get effect for regressor {name}: {e}")
                regressor_effects.append(0)
        
        plt.figure(figsize=(10, 5))
        plt.bar(regressors, regressor_effects, color='skyblue')
        plt.title('Regressor Effects on Innovation Levels', fontsize=14)
        plt.ylabel('Effect Size')
        plt.xlabel('Regressors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return prophet_results

# Example usage:
prophet_results = analyze_innovation_seasonality(
    cluster_results, 
    distance_type='internal',
    sample_size=0.01,
    bins=[0, 100],#1, 1.5, 2, 2.5, 
    forecast_years=10,
    cycle_years=45,
    seed=42,
    holidays=['1966-01-01', '1993-01-01']
)

# %% time series analysis
def time_series_analysis(cluster_data, artist_name=None, sample_size=0.1, seed=None,
                                   bins=[0, 1, 2, 3, 4], distance_type='historical',
                                   forecast_years=10, cycle_years=40, holidays=None, events=None):
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
        for event_year in events:
            df[f'year_since_{event_year}'] = (df['ds'].dt.year - event_year).clip(lower=0)
            df[f'streaming_{event_year}'] = (df['ds'] >= f'{event_year}-01-01').astype(int)
        
        # Step 4: Initialize Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            holidays_prior_scale=10.0,
        )
        
        # Add custom seasonality
        model.add_seasonality(
            name='long_cycle',
            period=cycle_years * 365.25,
            fourier_order=3
        )
        
        # Add regressors for intercept and slope changes
        for event_year in [1999, 2015]:
            model.add_regressor(f'streaming_{event_year}')
            model.add_regressor(f'year_since_{event_year}')
        
        # Step 5: Fit the model
        model.fit(df)
        
        # Step 6: Make future predictions
        future = model.make_future_dataframe(periods=forecast_years, freq='Y')
        for event_year in [1999, 2015]:
            future[f'year_since_{event_year}'] = (future['ds'].dt.year - event_year).clip(lower=0)
            future[f'streaming_{event_year}'] = (future['ds'] >= f'{event_year}-01-01').astype(int)
        
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
        
        # Plot regressor effects
        regressors = [f'streaming_{event_year}' for event_year in events] + [f'year_since_{event_year}' for event_year in events]
        regressor_effects = []
        
        # Safely access regressor coefficients from model parameters
        for name in regressors:
            try:
                # Convert params['beta'] to a dictionary if it's a Series/DataFrame
                beta_params = dict(zip(model.extra_regressors.keys(), 
                                     model.params['beta'].flatten()))
                effect = beta_params.get(name, 0)
                regressor_effects.append(effect)
            except Exception as e:
                print(f"Warning: Could not get effect for regressor {name}: {e}")
                regressor_effects.append(0)
        
        plt.figure(figsize=(10, 5))
        plt.bar(regressors, regressor_effects, color='skyblue')
        plt.title('Regressor Effects on Innovation Levels', fontsize=14)
        plt.ylabel('Effect Size')
        plt.xlabel('Regressors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return prophet_results

# Example usage:
prophet_results = time_series_analysis(
    cluster_results, 
    distance_type='historical',#historical, internal
    sample_size=0.3,
    bins=[0,1,2,3,100],#0,0.5,1, 1.5, 2, 2.5, 100
    forecast_years=0,
    cycle_years=40,#40
    holidays=['1966-01-01', '1993-01-01'],
    events=[1999, 2015]
)

#2个标准差外的
# %%
#不同参数：长周期；傅立叶阶数
#检验

#更大比例；不同距离；seed

#拼图


#年限
#