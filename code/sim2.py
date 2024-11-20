
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

#筛选指定的音乐人
df = df.filter(F.col('artist').isin(['Coldplay', 'Taylor Swift', 'Ed Sheeran', 'Ariana Grande', 'Avril Lavigne','Radiohead','The Beatles','Queen','Oasis','Bruno Mars','Lady Gaga','Katy Perry','Blur','The Rolling Stones','David Bowie','Michael Jackson']))

# Note potentially relevant features like danceability, energy, acousticness, etc.
df.columns

# %%
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
def find_optimal_kmeans(features, k_values=range(2, 11)):
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

    # find optimal k
    optimal_k = k_values[silhouettes.index(max(silhouettes))]
    print(f"\nOptimal number of clusters (k) = {optimal_k}")
    print(f"Best silhouette score = {max(silhouettes)}")
    
    # train final model with optimal k
    kmeans = KMeans(k=optimal_k, seed=1)
    kmeans_model = kmeans.fit(features)
    optimal_predictions = kmeans_model.transform(features)
    
    return optimal_k, optimal_predictions, silhouettes
# %% Basic KMeans
optimal_k_basic, kmeans_predictions_basic, silhouettes_basic = find_optimal_kmeans(features)

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


    print(f"Optimal number of components: {optimal_n}, threshold: {threshold}")

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
#%%
# 2. KMeans: find optimal k, based on PCA-transformed features
features_pca.persist()
optimal_k_pca, kmeans_predictions_pca, silhouettes_pca = find_optimal_kmeans(features_pca)
# %% SVD-KMeans
# for SVD, convert original features to RDD
def find_optimal_svd_components(df_features,threshold=0.9):
    """
    Find optimal number of SVD components using explained variance analysis
    
    Args:
        df_features: DataFrame with feature vectors
        
    Returns:
        n_components: Optimal number of components
        U_df: DataFrame with SVD-transformed features
        explained_variance_ratio: Array of explained variance ratios
        cumulative_variance_ratio: Array of cumulative explained variance ratios
    """
    # convert to RDD
    vectors_rdd = df_features.rdd.map(lambda row: row["features"])

    # use RDD-specific standardizer to re-scale data
    standardizer_rdd = StandardScalerRDD()
    model = standardizer_rdd.fit(vectors_rdd)
    vectors_rdd = model.transform(vectors_rdd)
    mat = RowMatrix(vectors_rdd)

    # Compute SVD without predefining the number of components
    svd = mat.computeSVD(mat.numCols(), computeU=True)

    # Access SVD components
    U = svd.U
    s = svd.s
    V = svd.V

    # convert U to DataFrame (and persist to memory)
    U_df = U.rows.map(lambda row: Row(features=Vectors.dense(row.toArray()))) \
                 .toDF()
    U_df.persist()

    s_values = svd.s.toArray()

    # Calculate explained variance ratio
    total_variance = (s_values ** 2).sum()
    explained_variance_ratio = (s_values ** 2) / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Print variance explained by each component
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
        print(f"Component {i+1}:")
        print(f"  Explained variance ratio: {var:.4f}")
        print(f"  Cumulative explained variance: {cum_var:.4f}")



    # Find number of components needed for 90% variance
    optimal_n = np.argmax(cumulative_variance_ratio >= threshold) + 1
    print(f"\nNumber of components {optimal_n} for {threshold} threshold")


    # Visualize singular values and explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scree plot
    ax1.plot(range(1, len(s_values) + 1), s_values, 'bo-')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Scree Plot')
    ax1.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal number of components ({optimal_n})')
    # Cumulative explained variance plot
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio * 100, 'bo-')
    ax2.axhline(y=threshold*100, color='r', linestyle='--', label=f'{threshold} threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()

    plt.tight_layout()
    plt.show()


    # SVD filtering
    # Select only the first n_components columns from U
    U_filtered = U.rows.map(lambda row: Vectors.dense(row.toArray()[:optimal_n]))

    # Convert U_filtered to DataFrame and persist to memory
    U_df = spark.createDataFrame(U_filtered.map(Row), ["features"])
    U_df.persist()

    # Note: we've reduced our dimensionality down to n_components dimensions
    U_df.show(5, truncate=False)

    return optimal_n, U_df, explained_variance_ratio, cumulative_variance_ratio
# 1. SVD: find optimal number of components
optimal_n_svd, features_svd, explained_variance_ratio, cumulative_variance_ratio = find_optimal_svd_components(df_features,threshold=0.9)
# %%
# 2. check dimensionality reduction according to V
# %%
# 3. KMeans: find optimal k, based on SVD-transformed features
features_svd.persist()
optimal_k_svd, kmeans_predictions_svd, silhouettes_svd = find_optimal_kmeans(features_svd)
# %% visualize clusters for a specific artist
def analyze_artist_clusters(predictions_df, original_df, artist_name, num_clusters,dim_1,dim_2):
    """
    Analyze and visualize clustering results for a specific artist
    
    Args:
        predictions_df: DataFrame with clustering predictions
        original_df: Original DataFrame with all song information
        artist_name: Name of the artist to analyze
        dim_1: index of the first dimension to plot
        dim_2: index of the second dimension to plot
    """
    predictions_df.groupby('prediction') \
               .count() \
               .show()
    df_merged = predictions_df.withColumn("id", F.monotonically_increasing_id()).withColumnRenamed("features", "raw_features") \
            .join(original_df.withColumn("id", F.monotonically_increasing_id()), on="id", how="inner") 
    
    results = df_merged.join(df.select  ('release_date', 'name', 'artist'), ['name', 'artist']) \
            .filter(F.col('artist') == artist_name)
        
    # Convert to pandas for easier plotting
    artist_data = results.toPandas()
    
    # Convert release_date to datetime
    artist_data['release_date'] = pd.to_datetime(artist_data['release_date'])
    artist_data['year'] = artist_data['release_date'].dt.year
    
    # Create subplots: general description and evolution over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Count of songs in each cluster
    cluster_counts = artist_data['prediction'].value_counts()
    ax1.bar(cluster_counts.index, cluster_counts.values)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Songs')
    ax1.set_title(f'Distribution of {artist_name} Songs Across Clusters')
    
    # Plot 2: Cluster distribution over time
    # Calculate proportion of each cluster per year
    yearly_proportions = artist_data.pivot_table(
        index='year',
        columns='prediction',
        aggfunc='size',
        fill_value=0
    )
    yearly_proportions = yearly_proportions.div(yearly_proportions.sum(axis=1), axis=0)
    
    # Plot stacked area chart
    yearly_proportions.plot(
        kind='area',
        stacked=True,
        ax=ax2,
        alpha=0.7
    )
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Proportion of Songs')
    ax2.set_title(f'Evolution of {artist_name} Song Clusters Over Time')
    
    plt.tight_layout()
    plt.show()

    # details: scatter plot of clusters in PCA space
    plt.figure(figsize=(10, 8))

    # Define markers and colors for each cluster
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h']  # Add more markers if needed
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))  # Use colormap for distinct colors

    artist_matrix = df_merged.filter(F.col("artist") == artist_name) \
                          .select("name", "prediction", "features") \
                          .toPandas()

    # Plot each cluster with different marker and color
    for cluster in artist_matrix["prediction"].unique():
        mask = artist_matrix["prediction"] == cluster
        plt.scatter(artist_matrix[mask]["features"].apply(lambda x: float(x[dim_1])),
                   artist_matrix[mask]["features"].apply(lambda x: float(x[dim_2])),
                   c=[colors[cluster]],
                   marker=markers[cluster % len(markers)],
                   label=f'Cluster {cluster}',
                   s=100)  # increase marker size

    # Add labels with adjustable positions to avoid overlap
    for i, txt in enumerate(artist_matrix["name"]):
        x = float(artist_matrix["features"].iloc[i][dim_1])
        y = float(artist_matrix["features"].iloc[i][dim_2])
        plt.annotate(txt, (x, y),
                xytext=(5, 5),  # 5 points offset
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8)

    plt.xlabel(f"Component {dim_1}")
    plt.ylabel(f"Component {dim_2}")
    plt.title(f"{artist_name} Songs Clustered in Dimensionally Reduced Space")
    plt.legend()
    plt.show()

    # Print summary statistics
    print(f"\nSummary for {artist_name}:")
    print(f"Total number of songs: {len(artist_data)}")
    print("\nSongs per cluster:")
    print(cluster_counts)
    
    return artist_data
#demo
artist_name="Coldplay"
analyze_artist_clusters(kmeans_predictions_pca, df_features, artist_name,num_clusters=optimal_k_pca,dim_1=0,dim_2=1)
# %%
#统一颜色
#即使没有也要显示在bar图

#调整参数

#按照factor贡献来呈现最终的图；不同的初始feature

#奇怪的异常值让图变得奇怪

#不同的variable
#不同的sample: 不同音乐类型的区别

# %%
