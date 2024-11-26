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

# 仅仅要release_date在1920-2020的
#df = df.filter(~(F.col("release_date").between("2017-01-01", "2020-12-31")))
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

#%% sample PCA illustration

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

#%% filter PCA features and merge with original data
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors

def filter_pca_features(features_pca, df, range_min=-10, range_max=10):
    """
    Filter PCA features and merge with original data
    """
    # Convert vector to array for more efficient filtering
    features_with_arrays = features_pca.select(
        'features',
        vector_to_array('features').alias('features_array')
    )
    
    # Add row index for joining
    features_with_arrays = features_with_arrays.withColumn("row_idx", F.monotonically_increasing_id())
    df_with_idx = df.withColumn("row_idx", F.monotonically_increasing_id())
    
    # Join PCA features with original data
    joined_df = features_with_arrays.join(df_with_idx, "row_idx")
    
    # Apply filter conditions using array operations
    filtered_features = joined_df.filter(
        (F.array_min('features_array') >= range_min) &
        (F.array_max('features_array') <= range_max)
    ).select('features', '*')
    
    return filtered_features

filtered_features = filter_pca_features(features_pca, df, -10, 10)
print(f"Number of songs before filtering: {features_pca.count()}")
print(f"Number of songs after filtering: {filtered_features.count()}")
filtered_features.show()

#%% KMeans
def find_optimal_kmeans(data, k_values):
    """
    Optimized K-means clustering for distributed computing
    """
    silhouettes = []
    
    # Cache the data to avoid recomputation
    data.cache()
    
    # Calculate sample size once
    total_count = data.count()
    sample_size = max(min(total_count * 0.1, 10000), 1000)  # Cap sample size
    
    for k in k_values:
        # Use stratified sampling for better representation
        sample_data = data.sample(
            withReplacement=False, 
            fraction=sample_size/total_count, 
            seed=42
        ).cache()
        
        # Configure K-means for better distributed performance
        kmeans = (KMeans()
                 .setK(k)
                 .setMaxIter(20)
                 .setTol(1e-4)
                 .setSeed(42)
                 .setDistanceMeasure("euclidean")  # Explicitly set distance measure
                 .setInitMode("k-means||"))        # Use k-means|| initialization
        
        # Fit model with error handling
        try:
            model = kmeans.fit(data)
            predictions = model.transform(sample_data)
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            silhouettes.append((k, silhouette))
            print(f"Silhouette score for k={k}: {silhouette}")
        except Exception as e:
            print(f"Error processing k={k}: {str(e)}")
            silhouettes.append((k, -1))  # Mark failed attempts
        
        # Clean up to free memory
        sample_data.unpersist()
    
    # Clean up cached data
    data.unpersist()
    
    # Filter out failed attempts and find optimal k
    valid_silhouettes = [(k, s) for k, s in silhouettes if s >= 0]
    optimal_k = max(valid_silhouettes, key=lambda x: x[1])[0] if valid_silhouettes else k_values[0]

    
    return optimal_k, silhouettes, final_predictions

# Execution with optimized parameters
filtered_features = filter_pca_features(features_pca, -10, 10).cache()
print(f"Number of songs after filtering: {filtered_features.count()}")

# Adjust k_values range based on data size
k_values = range(2, min(4, int(filtered_features.count() ** 0.5)))
optimal_k, silhouettes, final_predictions = find_optimal_kmeans(filtered_features, k_values)
print(f"Optimal number of clusters: {optimal_k}")


#%% DBSCAN
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def visualize_dbscan_clusters(raw_features, eps=0.5, min_samples=5, sample_size=0.00001):
    """
    Perform DBSCAN clustering and visualize results using PCA components.
    
    Args:
        features: Spark DataFrame with PCA features
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        sample_size: Fraction of data to sample for visualization
    
    Returns:
        None (displays plot)
    """
    # Sample features for better performance
    features = raw_features.sample(withReplacement=False, fraction=sample_size, seed=42)
    print(f"Number of songs after sampling: {features.count()}")
    
    try:
        # Cache features for better performance
        features.cache()
        
        # Standardize features
        standardizer = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        scaler = standardizer.fit(features)
        scaled_features = scaler.transform(features)
        
        # Convert vector to array first
        scaled_features = scaled_features.select(
            vector_to_array("scaled_features").alias("features_array")
        )
        
        # Convert to Pandas for DBSCAN computation
        pandas_df = scaled_features.select(
            F.col("features_array")[0].alias("x"),
            F.col("features_array")[1].alias("y")
        ).toPandas()
        
        # Extract features as NumPy array for DBSCAN
        X = pandas_df[['x', 'y']].values
        
        # 自动寻找最优DBSCAN参数组合
        def find_optimal_dbscan_params(filtered_features):
            # 定义参数搜索范围
            from sklearn.metrics import silhouette_score
            eps_range = np.arange(0.1, 0.2, 0.05)
            min_samples_range = [10, 20, 30]
            
            best_score = -1
            best_params = None
            results = []
            
            for eps in eps_range:
                for min_samples in min_samples_range:
                    print(f"\nTrying eps={eps}, min_samples={min_samples}")
                    
                    # 对当前参数组合进行DBSCAN聚类
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                    labels = dbscan.fit_predict(X)
                    
                    # 计算轮廓系数作为评估指标
                    if len(np.unique(labels)) > 1:  # 至少有两个簇(不包括噪声点)
                        silhouette_avg = silhouette_score(X, labels)
                        results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'score': silhouette_avg,
                            'n_clusters': len(np.unique(labels[labels != -1])),
                            'noise_ratio': np.sum(labels == -1) / len(labels)
                        })
                        print(f"Silhouette Score: {silhouette_avg:.3f}")
                        print(f"Number of clusters: {len(np.unique(labels[labels != -1]))}")
                        print(f"Noise ratio: {np.sum(labels == -1) / len(labels):.3f}")
                        
                        # 更新最优参数
                        if silhouette_avg > best_score:
                            best_score = silhouette_avg
                            best_params = {'eps': eps, 'min_samples': min_samples}
            
            # 可视化参数搜索结果
            results_df = pd.DataFrame(results)
            plt.figure(figsize=(12, 5))
            
            plt.subplot(121)
            plt.scatter(results_df['eps'], results_df['score'])
            plt.xlabel('eps')
            plt.ylabel('Silhouette Score')
            plt.title('Parameter Search Results')
            
            plt.subplot(122)
            plt.scatter(results_df['min_samples'], results_df['score'])
            plt.xlabel('min_samples')
            plt.ylabel('Silhouette Score')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nBest parameters found:")
            print(f"eps: {best_params['eps']}")
            print(f"min_samples: {best_params['min_samples']}")
            print(f"Best silhouette score: {best_score:.3f}")
            
            return best_params

        # 执行参数优化
        #optimal_params = find_optimal_dbscan_params(filtered_features)

        # Perform DBSCAN clustering
        #dbscan = DBSCAN(eps=optimal_params['eps'], min_samples=optimal_params['min_samples'], metric='euclidean')
        dbscan = DBSCAN(eps=0.05, min_samples=30, metric='euclidean')
        pandas_df['cluster'] = dbscan.fit_predict(X)
        
        
        # Visualize results
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            pandas_df['x'], 
            pandas_df['y'], 
            c=pandas_df['cluster'], 
            cmap='viridis', 
            marker='o', 
            alpha=0.6
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f'DBSCAN Visualization (eps={eps}, min_samples={min_samples})')
        plt.grid(True, alpha=0.3)
        
        # Add legend for noise points
        unique_clusters = pandas_df['cluster'].unique()
        legend_elements = []
        for cluster in unique_clusters:
            color = scatter.cmap(scatter.norm(cluster))
            if cluster == -1:  # Noise points
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='Noise', markersize=10)
                )
            else:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=f'Cluster {cluster}', markersize=10)
                )
        plt.legend(handles=legend_elements)
        
        plt.show()
        
    except Exception as e:
        print(f"Error in DBSCAN clustering or visualization: {str(e)}")
        
    finally:
        # Clean up
        features.unpersist()
    
visualize_dbscan_clusters(filtered_features, eps=0.4, min_samples=50, sample_size=0.01)
# %% Hierarchical Clustering
from pyspark.ml.feature import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator
def visualize_hierarchical_clusters(raw_features, linkage_method="ward", sample_size=0.01, n_clusters=None):
    """
    Perform Hierarchical Clustering and visualize results using PCA components.
    Automatically determine optimal number of clusters using elbow method if n_clusters is not specified.
    
    Args:
        raw_features: Spark DataFrame with PCA features
        linkage_method: Linkage method for hierarchical clustering ("ward", "complete", "average", "single")
        sample_size: Fraction of data to sample for visualization
        n_clusters: Number of clusters to form. If None, will determine automatically
    
    Returns:
        pandas_df: Pandas DataFrame with cluster assignments
        n_clusters: Number of clusters used
    """
    # Sample features for better performance
    features = raw_features.sample(withReplacement=False, fraction=sample_size, seed=42)
    print(f"Number of songs after sampling: {features.count()}")
    
    try:
        # Cache features for better performance
        features.cache()
        
        # Standardize features
        standardizer = StandardScaler(inputCol="features", outputCol="scaled_features")
        scaler = standardizer.fit(features)
        scaled_features = scaler.transform(features)
        
        # Convert vector to array
        scaled_features = scaled_features.select(
            vector_to_array("scaled_features").alias("features_array")
        )
        
        # Convert to Pandas for hierarchical clustering
        pandas_df = scaled_features.select(
            F.col("features_array")[0].alias("x"),
            F.col("features_array")[1].alias("y")
        ).toPandas()
        
        # Extract features as NumPy array
        X = pandas_df[['x', 'y']].values
        
        # Perform hierarchical clustering
        Z = linkage(X, method=linkage_method)  # Compute the linkage matrix
        
        if n_clusters is None:
            # Find optimal number of clusters using distance differences
            last_distances = Z[:, 2]
            distance_diffs = np.diff(last_distances)
            max_diff_idx = np.argmax(distance_diffs)
            optimal_clusters = len(X) - max_diff_idx
            print(f"Optimal number of clusters based on maximum distance difference: {optimal_clusters}")
            n_clusters = optimal_clusters
        else:
            print(f"Using specified number of clusters: {n_clusters}")
            
        # Form clusters using optimal/specified number
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        # Add cluster labels to the dataframe
        pandas_df['cluster'] = cluster_labels
        
        # Visualize dendrogram with distance differences
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90., leaf_font_size=10.)
        plt.title(f"Dendrogram ({linkage_method} linkage)")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        
        # Plot distance differences
        plt.subplot(2, 1, 2)
        distance_diffs = np.diff(Z[:, 2])
        plt.plot(range(len(distance_diffs)), distance_diffs, 'b-')
        if n_clusters is None:
            if optimal_clusters == n_clusters:
                plt.axvline(x=max_diff_idx, color='r', linestyle='--', 
                           label=f'Maximum difference at {max_diff_idx}')
        plt.title("Distance Differences Between Merges")
        plt.xlabel("Merge Step")
        plt.ylabel("Distance Difference")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Visualize clusters in 2D space
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            pandas_df['x'], 
            pandas_df['y'], 
            c=pandas_df['cluster'], 
            cmap='viridis', 
            marker='o', 
            alpha=0.6
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f"Hierarchical Clustering Visualization\n({linkage_method} linkage, {n_clusters} clusters)")
        plt.grid(True, alpha=0.3)
        
        # Add legend for clusters
        unique_clusters = np.unique(cluster_labels)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=scatter.cmap(scatter.norm(i)),
                       label=f'Cluster {i}', markersize=10)
            for i in unique_clusters
        ]
        plt.legend(handles=legend_elements)
        plt.show()
        
        return pandas_df, n_clusters
        
    except Exception as e:
        print(f"Error in hierarchical clustering or visualization: {str(e)}")
        return None, None
        
    finally:
        # Clean up
        features.unpersist()


#%% Hierarchical Clustering example
visualize_hierarchical_clusters(filtered_features, linkage_method="ward", sample_size=0.0001)
visualize_hierarchical_clusters(filtered_features, linkage_method="ward", sample_size=0.0001, n_clusters=4)

# %% Hierarchical Clustering comparison
def compare_clustering_methods(filtered_features,
                             linkage_methods=['ward', 'complete', 'average'],
                             sample_size=0.0001, n_clusters=None):
    """
    Compare different linkage methods by running visualizations
    and storing results for comparison.
    
    Args:
        filtered_features: Input features for clustering
        linkage_methods: List of linkage methods to try
        sample_size: Sample size for visualization
        n_clusters: Number of clusters (optional)
    """
    # Store results for comparison
    results = []
    
    try:
        for method in linkage_methods:
            print(f"\nTesting linkage method: {method}")
            
            # Run visualization with current parameters
            print(f"Running {method} method...")
            visualize_hierarchical_clusters(
                filtered_features, 
                linkage_method=method,
                sample_size=sample_size,
                n_clusters=n_clusters
            )
                
    except Exception as e:
        print(f"Error during comparison: {str(e)}")

# Example usage
compare_clustering_methods(
    filtered_features,
    sample_size=0.0001,
    n_clusters=5
)
# %% Hierarchical Clustering comparison example
compare_clustering_methods(
    filtered_features,
    sample_size=0.1
)
compare_clustering_methods(
    filtered_features,
    sample_size=0.1,
    n_clusters=4
)

# %%
#音乐家 时间 雷达
#联系
#时间
#spark

# %% 匹配数据

# %%
standardizer = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler = standardizer.fit(filtered_features)
scaled_features = scaler.transform(filtered_features)

#%%

from pyspark.sql import functions as F
from pyspark.ml.feature import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pyspark.sql import Row

def visualize_hierarchical_clusters_with_results(raw_features, linkage_method="ward", sample_size=0.01, n_clusters=None):
    """
    Perform Hierarchical Clustering and visualize results using PCA components.
    Automatically determine the optimal number of clusters using the distance method if n_clusters is not specified.
    
    Args:
        raw_features: Spark DataFrame with PCA features
        linkage_method: Linkage method for hierarchical clustering ("ward", "complete", "average", "single")
        sample_size: Fraction of data to sample for visualization
        n_clusters: Number of clusters to form. If None, will determine automatically
    
    Returns:
        updated_features: Spark DataFrame with cluster assignments
        n_clusters: Number of clusters used
    """
    # Sample features for better performance
    features = raw_features.sample(withReplacement=False, fraction=sample_size, seed=42)
    num_samples = features.count()
    if num_samples > 10000:#二次采样
        features = features.sample(withReplacement=False, fraction=1000/num_samples, seed=42)

    print(f"Number of songs after sampling: {features.count()}")
    
    # Cache features for better performance
    features.cache()
    
    try:
        
        # Convert vector to array
        scaled_features = features.select(
            "row_idx",  # Ensure we keep a reference to the original rows
            vector_to_array("scaled_features").alias("features_array")
        )
        
        # Convert to Pandas for hierarchical clustering
        pandas_df = scaled_features.select(
            "row_idx",
            F.col("features_array")[0].alias("x"),
            F.col("features_array")[1].alias("y")
        ).toPandas()
        
        # Extract features as NumPy array
        X = pandas_df[['x', 'y']].values
        
        # Perform hierarchical clustering
        Z = linkage(X, method=linkage_method)  # Compute the linkage matrix
        
        # Find optimal number of clusters using distance threshold if not specified
        distances = Z[:, 2]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_dist
        
        if n_clusters is None:
            # Use distance threshold to determine clusters
            cluster_labels = fcluster(Z, t=threshold, criterion='distance')
            n_clusters = len(np.unique(cluster_labels))
            print(f"Optimal number of clusters based on distance threshold: {n_clusters}")
            use_threshold = True
        else:
            # Use specified number of clusters
            print(f"Using specified number of clusters: {n_clusters}")
            cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
            use_threshold = False
            
        # Add cluster labels to the dataframe
        pandas_df['cluster'] = cluster_labels
        
        # Visualize dendrogram with distance threshold
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=90., leaf_font_size=10.)
        if use_threshold:
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Distance threshold: {threshold:.2f}')
            plt.legend()
        plt.title(f"Dendrogram ({linkage_method} linkage)")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        
        # Plot distance distribution
        plt.subplot(2, 1, 2)
        plt.hist(distances, bins=50)
        if use_threshold:
            plt.axvline(x=threshold, color='r', linestyle='--', 
                       label=f'Distance threshold: {threshold:.2f}')
            plt.legend()
        plt.title("Distribution of Merge Distances")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        
        # Visualize clusters in 2D space
        plt.figure(figsize=(12, 8))
        if features.count() > 10000:
            alpha = 0.06
        else:
            alpha = 0.6
        scatter = plt.scatter(
            pandas_df['x'], 
            pandas_df['y'], 
            c=pandas_df['cluster'], 
            cmap='viridis', 
            marker='o', 
            alpha=alpha
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f"Hierarchical Clustering Visualization\n({linkage_method} linkage, {n_clusters} clusters)")
        plt.grid(True, alpha=0.3)
        
        # Add legend for clusters
        unique_clusters = np.unique(cluster_labels)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=scatter.cmap(scatter.norm(i)),
                       label=f'Cluster {i}', markersize=10)
            for i in unique_clusters
        ]
        plt.legend(handles=legend_elements)
        plt.show()
        
        # Map cluster results back to the original Spark DataFrame
        cluster_mapping = pandas_df[['row_idx', 'cluster']].apply(
            lambda row: Row(row_idx=row['row_idx'], cluster=int(row['cluster'])),
            axis=1
        )
        cluster_mapping_df = spark.createDataFrame(cluster_mapping.tolist())
        
        # Join cluster labels back to the original DataFrame
        updated_features = raw_features.join(cluster_mapping_df, on='row_idx', how='left')
        
        return updated_features, n_clusters
        
    except Exception as e:
        return None, None
        #print(f"Error in hierarchical clustering or visualization: {str(e)}")
        #raise e  # Re-raise the exception to see the full error trace
        
    finally:
        # Clean up
        features.unpersist()



# %% year or artist
def cluster_by_filter(features, filter_type='year', filter_value=2020, 
                     linkage_method="ward", sample_size=0.99, n_clusters=None):
    """
    Perform hierarchical clustering on filtered data by year or artist
    
    Args:
        features: Input features DataFrame
        filter_type: 'year' or 'artist' to filter by
        filter_value: Year (int) or artist name (str) to filter for
        linkage_method: Linkage method for hierarchical clustering
        sample_size: Fraction of data to sample
        n_clusters: Number of clusters (optional)
    """

    # Standardize features at global level
    

    if filter_type == 'year':
        filtered_df = features.filter(F.year(F.col("release_date")) == filter_value)
    elif filter_type == 'artist':
        filtered_df = features.filter(F.col("artist") == filter_value)
    else:
        raise ValueError("filter_type must be 'year' or 'artist'")
        
    
    return visualize_hierarchical_clusters_with_results(
        raw_features=filtered_df,
        linkage_method=linkage_method,
        sample_size=sample_size,
        n_clusters=n_clusters
    )

# Example usage for year 2020
for linkage_method in ["ward"]:
    clustered_data, num_clusters = cluster_by_filter(
        scaled_features,
        linkage_method=linkage_method,
        filter_type='artist', 
        filter_value='Coldplay',
        n_clusters=3
    )

