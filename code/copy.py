#%% nodes=2 60G seed=21; 起始点很重要，而这些配置会影响起始点
# 2. KMeans: find optimal k, based on PCA-transformed features
#seed = 11
#optimal_k_pca, kmeans_predictions_pca, silhouettes_pca = find_optimal_kmeans(features_pca, k_values=range(5, 6), seed=seed)
#print(f"Using random seed: {seed}")
#%%
'''
seed = 10
while True:
    optimal_k_pca, kmeans_predictions_pca, silhouettes_pca = find_optimal_kmeans(features_pca, k_values=[2,5], seed=seed)
    print(f"Using random seed: {seed}")
    if silhouettes_pca[1] > silhouettes_pca[0] :
        print(f"Found good seed {seed} with silhouette score {silhouettes_pca[0]}")
        break
    seed += 1
'''
#%% multiple seeds
#optimal_k_pca, kmeans_predictions_pca, silhouettes_pca, seeds = find_optimal_kmeans_multiple_seeds(features_pca, k_values=range(2, 13), n_seeds=5)

# %% hierarchical clustering
# Bisecting K-means clustering
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
def bisecting_kmeans_clustering(features_pca, k_values=range(2, 13), seed=42):
    """
    Perform bisecting k-means clustering on PCA-transformed features and find optimal k
    
    Args:
        features_pca: DataFrame with PCA-transformed features
        k_values: Range of k values to try
        
    Returns:
        optimal_predictions: DataFrame with cluster predictions using optimal k
        optimal_model: Fitted BisectingKMeans model with optimal k
        silhouettes: List of silhouette scores for each k
    """
    silhouettes = []
    models = {}
    predictions = {}
    
    # Try different k values
    for k in k_values:
        # Initialize and fit bisecting k-means model
        bkm = BisectingKMeans(k=k, seed=seed, featuresCol="features")
        model = bkm.fit(features_pca)
        
        # Make predictions
        pred = model.transform(features_pca)
        
        # Evaluate clustering using silhouette score
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(pred)
        silhouettes.append(silhouette)
        
        print(f"Silhouette score for k={k}: {silhouette}")
        
        models[k] = model
        predictions[k] = pred
    
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

    # Find optimal k
    optimal_k = k_values[np.argmax(silhouettes)]
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Print cluster centers for optimal model
    print("\nCluster Centers:")
    centers = models[optimal_k].clusterCenters()
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")
        
    return predictions[optimal_k], models[optimal_k], silhouettes

# Run bisecting k-means clustering
#print("Running Bisecting K-means clustering...")
#bisecting_predictions, bisecting_model = bisecting_kmeans_clustering(features_pca, seed=33)





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
#small_clusters = cluster_counts.filter(F.col("count") < 10000).select("prediction").rdd.flatMap(lambda x: x).collect()
#cluster_results = merged_results.filter(~F.col("prediction").isin(small_clusters))
cluster_results = merged_results

# filter out songs before 1920 and after 2020
cluster_results = cluster_results.filter(F.year(F.to_timestamp('release_date')).between(1920, 2020))

# old cluster distribution
cluster_results.groupby('prediction') \
              .count() \
              .orderBy('prediction') \
              .show()
'''

from pyspark.sql import Window
from pyspark.sql import functions as F

# get distinct cluster IDs and sort
distinct_clusters = cluster_results.select("prediction").distinct().orderBy("prediction")

# create new cluster IDs
window_spec = Window.orderBy("prediction")
renumbered_clusters = distinct_clusters.withColumn("new_prediction", F.row_number().over(window_spec) - 1)

# map back to original data
cluster_results = cluster_results.join(
    renumbered_clusters, on="prediction", how="left"
).drop("prediction").withColumnRenamed("new_prediction", "prediction")

cluster_results.groupby('prediction') \
              .count() \
              .orderBy('prediction') \
              .show()
'''

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

def plot_cluster_scatter(cluster_data, artist_name=None, sample_size=0.1, seed=None):
    """Plot scatter plot of clusters in PCA space with different dimension combinations"""
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Extract PCA coordinates
    pca_coords = np.vstack(cluster_data['features'].values)
    
    # Get dimensions to plot
    dims = [(0,1), (0,2), (1,2)]  # Different combinations of first 3 PCs
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cluster Distribution in PCA Space', fontsize=16)
    
    for ax, (dim1, dim2) in zip(axes, dims):
        # Plot each cluster
        for cluster in range(num_clusters):
            mask = cluster_data['prediction'] == cluster
            ax.scatter(pca_coords[mask, dim1], 
                      pca_coords[mask, dim2],
                      c=[colors[cluster]], 
                      label=f'Cluster {cluster}',
                      alpha=0.6,
                      s=20)
        
        ax.set_xlabel(f'PC{dim1+1}')
        ax.set_ylabel(f'PC{dim2+1}')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first subplot
        if dim1 == 0 and dim2 == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if artist_name:
        plt.suptitle(f'Cluster Distribution in PCA Space - {artist_name}', fontsize=16)
    
    plt.tight_layout()
    plt.show()

#%% show space and representative tracks: to filter out uninterested music
plot_cluster_scatter(cluster_results)

# Show top 10 songs from each cluster
for cluster_id in range(cluster_results.select('prediction').distinct().count()):
    print(f"\nTop 10 songs in Cluster {cluster_id}:")
    cluster_results.filter(F.col('prediction') == cluster_id) \
                  .select('name', 'artist', 'release_date') \
                  .show(10, truncate=False)
plot_cluster_radar(cluster_results)
plot_cluster_evolution(cluster_results)

# %%
# 将cluster2的id储存
story_id = [row.id for row in cluster_results.filter(F.col('prediction') == 2).select('id').collect()]
#%%
print(len(story_id))
# 保存story_id为csv
pd.DataFrame({'id': story_id, 'label': ['story'] * len(story_id)}).to_csv('remove_id.csv', index=False)



# %%
