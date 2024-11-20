# ## Scalable Dimension Reduction and Clustering with Spark

# %%
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
spark = SparkSession \
        .builder \
        .appName("dr_cluster") \
        .getOrCreate()

# Read Spotify data
df = spark.read.csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True)

# Note potentially relevant features like danceability, energy, acousticness, etc.
df.columns

# %%
'''
['id',
 'name',
 'popularity',
 'duration_ms',
 'explicit', #不要
 'release_date',
 'danceability',
 'energy',
 'key',
 'loudness',
 'mode',
 'speechiness',
 'acousticness',
 'instrumentalness',
 'liveness',
 'valence',
 'tempo',
 'time_signature',
 'artist']
 '''

# %% Data preprocessing
# identify potentially relevant features and add to a feature dataframe
feature_cols = ['duration_ms', 'loudness',
                'liveness',
                'key','tempo', 'time_signature','mode',
                'acousticness', 'instrumentalness',  'speechiness',
                'danceability', 'energy','valence']

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
silhouettes = []
k_values = range(2, 11)  # Try k from 2 to 10

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
kmeans_predictions = kmeans_model.transform(features)

# %% PCA
# fit model
# Fit PCA model without specifying k (use max possible components)
n_features = len(feature_cols)  # Get number of original features
pca = PCA(k=n_features, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features)

# Get explained variance ratio for each component
explained_variances = model.explainedVariance
cumulative_variance = [sum(explained_variances[:i+1]) for i in range(len(explained_variances))]

# Print variance explained by each component
for i, var in enumerate(explained_variances):
    print(f"Component {i+1} explained variance: {var:.4f}")
    print(f"Cumulative explained variance: {cumulative_variance[i]:.4f}")

# Note: Now we can visually inspect the explained variance ratios
# and choose k based on:
# 1. Cumulative explained variance reaching desired threshold (e.g. 0.8 or 0.9)
# 2. Individual components contributing minimal additional variance
# 3. Elbow in scree plot
# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot scree plot
ax1.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot')

# Plot cumulative variance
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Cumulative Explained Variance Ratio')
ax2.set_title('Cumulative Explained Variance Plot')

plt.tight_layout()
plt.show()

# %%
# For demonstration, let's use k=2 for visualization
# But you should adjust k based on the explained variance analysis above
# Replace the existing PCA code with:
pca_full = PCA(k=n_features, inputCol="features", outputCol="pcaFeatures")
model_full = pca_full.fit(features)

# Calculate cumulative explained variance ratio
explained_variances = model_full.explainedVariance
cumulative_variance = np.cumsum(explained_variances)

# Find number of components needed for desired variance (e.g. 90%)
target_variance = 0.90
ideal_components = np.argmax(cumulative_variance >= target_variance) + 1

# Create new PCA model with optimal number of components
pca = PCA(k=ideal_components, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features)

# transform feature data
print(f"Using {ideal_components} components")
pca_results = model.transform(features).select("pcaFeatures")
pca_features = pca_results.rdd.map(lambda row: Vectors.dense(row.pcaFeatures))
pca_features = spark.createDataFrame(pca_features.map(Row), ["features"])

# persist data before training model on PCA-discovered features
pca_features.persist()

# Show reduced features
pca_features.show(5, truncate=False)

# %%
# Try different k values and compare to choose the optimal one
k_values = range(2, 11)  # Try k from 2 to 10
silhouette_scores = []

for k in k_values:
    # Fit model
    pca_kmeans = KMeans(k=k, seed=1)
    pca_model = pca_kmeans.fit(pca_features)

    # Make predictions (i.e. identify clusters)
    pca_predictions = pca_model.transform(pca_features)

    # Evaluate clustering by computing silhouette coefficient
    pca_evaluator = ClusteringEvaluator()
    silhouette = pca_evaluator.evaluate(pca_predictions)
    silhouette_scores.append(silhouette)
    print(f"Silhouette score for k={k}: {silhouette}")

# Find the optimal k
optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters: {optimal_k}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()

#%%
# Fit the model with the optimal k
optimal_pca_kmeans = KMeans(k=optimal_k, seed=1)
optimal_pca_model = optimal_pca_kmeans.fit(pca_features)

# Make predictions with the optimal model
optimal_pca_predictions = optimal_pca_model.transform(pca_features)

# Evaluate the optimal clustering
optimal_pca_evaluator = ClusteringEvaluator()
optimal_pca_silhouette = optimal_pca_evaluator.evaluate(optimal_pca_predictions)
print(f"Silhouette score with optimal k={optimal_k}: {optimal_pca_silhouette}")


# %% SVD
# convert to RDD
vectors_rdd = df_features.rdd.map(lambda row: row["features"])

# use RDD-specific standardizer to re-scale data
standardizer_rdd = StandardScalerRDD()
model = standardizer_rdd.fit(vectors_rdd)
vectors_rdd = model.transform(vectors_rdd)
mat = RowMatrix(vectors_rdd)

# Compute SVD
# Compute SVD without predefining the number of components
svd = mat.computeSVD(mat.numCols(), computeU=True)

# Access SVD components
U = svd.U
s = svd.s
V = svd.V

# convert U to DataFrame (and persist to memory) for clustering with K-Means
U_df = U.rows.map(lambda row: Row(features=Vectors.dense(row.toArray()))) \
             .toDF()
U_df.persist()

# Note: we've reduced our dimensionality down to 2 dimensions again
U_df.show(5, truncate=False)
#%%
#SVD筛选维度
print(svd.s)
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

# Visualize singular values and explained variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Scree plot
ax1.plot(range(1, len(s_values) + 1), s_values, 'bo-')
ax1.set_xlabel('Component')
ax1.set_ylabel('Singular Value')
ax1.set_title('Scree Plot')

# Cumulative explained variance plot
ax2.plot(range(1, len(cumulative_variance_ratio) + 1), 
         cumulative_variance_ratio * 100, 'ro-')
ax2.axhline(y=90, color='g', linestyle='--', label='90% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance (%)')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()

plt.tight_layout()
plt.show()

# Find number of components needed for 90% variance
n_components = np.argmax(cumulative_variance_ratio >= 0.9) + 1
print(f"\nNumber of components needed for 90% variance: {n_components}")

# SVD filtering
# Select only the first n_components columns from U
U_filtered = U.rows.map(lambda row: Vectors.dense(row.toArray()[:n_components]))

# Convert U_filtered to DataFrame and persist to memory
U_df = spark.createDataFrame(U_filtered.map(Row), ["features"])
U_df.persist()

# Note: we've reduced our dimensionality down to n_components dimensions
U_df.show(5, truncate=False)


# %%
# train model
# Try different values of k and store results
silhouette_scores = []
for k in range(2, 8):
    svd_kmeans = KMeans(k=k, seed=1)
    model = svd_kmeans.fit(U_df)
    predictions = model.transform(U_df)
    evaluator = ClusteringEvaluator()
    score = evaluator.evaluate(predictions)
    silhouette_scores.append((k, score))
    print(f"K={k}, Silhouette Score={score:.4f}")

# Find best k
best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
print(f"\nBest K={best_k} with Silhouette Score={best_score:.4f}")

# Use best k for final model
svd_kmeans = KMeans(k=best_k, seed=1)
svd_model = svd_kmeans.fit(U_df)

# make predictions (i.e. identify clusters)
svd_predictions = svd_model.transform(U_df)

# evaluate clustering by computing silhouette score
svd_evaluator = ClusteringEvaluator()
svd_silhouette = svd_evaluator.evaluate(svd_predictions)
print("Silhouette with squared euclidean distance = " + str(svd_silhouette))
#large value of silhouette score indicates good clustering (from -1 to 1)

# %%
predictions=optimal_pca_predictions
predictions.groupby('prediction') \
               .count() \
               .show()

df_features_with_id = df_features.withColumn("id", F.monotonically_increasing_id())
predictions_with_id = predictions.withColumn("id", F.monotonically_increasing_id()) \
                                       .withColumnRenamed("features", "adjusted_features")

# Perform an inner join on the 'id' column to merge df_features with U_df 
df_merged = df_features_with_id.join(predictions_with_id, on="id", how="inner")
df_merged.show(5)


# %% specific artist    ##################
artist_name="Coldplay"  
# Convert to pandas for plotting
artist_matrix = df_merged.filter(F.col("artist") == artist_name) \
                          .select("name", "prediction", "features") \
                          .toPandas()

artist_matrix["Component 1"] = artist_matrix["features"].apply(lambda x: float(x[0]))
artist_matrix["Component 2"] = artist_matrix["features"].apply(lambda x: float(x[1]))

# Convert to pandas for analysis
artist_clusters = df_merged.filter(F.col("artist") == artist_name) \
                         .select("name", "artist", "prediction") \
                         .toPandas()
#print(album_clusters)
date_mapping=df.filter(F.col("artist") == artist_name).select("name",  "release_date").toPandas()
artist_clusters=artist_clusters.merge(date_mapping, on="name", how="left")
artist_clusters.groupby(["release_date", "prediction"]) \
           .count() \
           .reset_index() \
           .rename(columns={"count": "track_count"}) \
           .sort_values(by="release_date")
print(artist_clusters)

# Create a pivot table for the heatmap
heatmap_data = artist_clusters.groupby(["release_date", "prediction"]).size().unstack(fill_value=0)
# Calculate proportions within each album
num_clusters = optimal_k
heatmap_data = heatmap_data.reindex(columns=range(num_clusters), fill_value=0)  # Ensure all clusters exist
heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
heatmap_data.to_csv("heatmap_data.csv")
print(heatmap_data)
# %%
from matplotlib import pyplot as plt
# Create scatter plot
plt.figure(figsize=(10, 8))

# Define markers and colors for each cluster
markers = ['o', 's', '^', 'v', 'D', 'p', 'h']  # Add more markers if needed
colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))  # Use colormap for distinct colors

# Plot each cluster with different marker and color
for cluster in artist_matrix["prediction"].unique():
    mask = artist_matrix["prediction"] == cluster
    plt.scatter(artist_matrix[mask]["Component 1"],
               artist_matrix[mask]["Component 2"], 
               c=[colors[cluster]],
               marker=markers[cluster % len(markers)],
               label=f'Cluster {cluster}',
               s=100)  # increase marker size

# Add labels with adjustable positions to avoid overlap
for i, txt in enumerate(artist_matrix["name"]):
    x = artist_matrix["Component 1"].iloc[i]
    y = artist_matrix["Component 2"].iloc[i]
    plt.annotate(txt, (x, y),
                xytext=(5, 5),  # 5 points offset
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title(f"{artist_name} Songs Clustered in PCA Space")
plt.legend()
plt.show()

# Create a heatmap using matplotlib
plt.figure(figsize=(12, 8))
im = plt.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd')

# Set x and y axis labels
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

# Add colorbar
plt.colorbar(im)

# Add title and labels
plt.title(f"Cluster Distribution Over Time for {artist_name} Songs")
plt.xlabel("Cluster")
plt.ylabel("Release Date")

# Add text annotations showing the values
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        text = plt.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                       ha='center', va='center',
                       color='black' if heatmap_data.values[i, j] < 0.5 else 'white')

plt.tight_layout()
plt.show()



# %%
