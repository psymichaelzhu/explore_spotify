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
df_raw = spark.read.csv('/home/mikezhu/data/spotify_data.csv', header=True)

# Note potentially relevant features like danceability, energy, acousticness, etc.
print(df_raw.columns)
print("Number of rows before filtering:", df_raw.count())
df_raw.show(5, truncate=False)
# %%
col_names=['track_id', 'streams', 'artist_followers', 'genres', 'album_total_tracks', 'track_artists', 'artist_popularity', 'explicit', 'tempo', 'chart', 'album_release_date', 'energy', 'key', 'added_at', 'popularity', 'track_album_album', 'duration_ms', 'available_markets', 'track_track_number', 'rank', 'mode', 'time_signature', 'album_name', 'speechiness', 'region', 'danceability', 'valence', 'acousticness', 'liveness', 'trend', 'instrumentalness', 'loudness', 'name']

# %%
# Extract and process genres
def extract_genres(genres_str):
    # Remove brackets and single quotes, then split by commas
    if genres_str and genres_str != '[]':
        # Remove brackets
        genres_str = genres_str.strip('[]')
        # Split by comma and clean up each genre
        return [g.strip().strip("'") for g in genres_str.split(',')]
    return []

# Create a new DataFrame with exploded genres
genres_df = df_raw.select('genres').rdd \
    .flatMap(lambda x: extract_genres(x.genres)) \
    .map(lambda x: Row(genre=x)) \
    .distinct()

genres_df = spark.createDataFrame(genres_df)

# Show total number of unique genres and list them
print(f"Total number of unique genres: {genres_df.count()}")
genres_df.show(truncate=False)
# Optional: Save to CSV if needed
# genres_df.toPandas().to_csv('unique_genres.csv', index=False)

# %%
genres_df.count()
# %%
# Count number of songs per genre
def count_songs_per_genre(df):
    """
    Count number of songs for each genre in the dataset
    
    Args:
        df: Spark DataFrame containing genres column
    Returns:
        DataFrame with genre counts sorted in descending order
    """
    # Extract and explode genres
    genre_counts = df.select('genres').rdd \
        .flatMap(lambda x: extract_genres(x.genres)) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: Row(genre=x[0], count=x[1]))
    
    # Convert to DataFrame and sort
    genre_counts_df = spark.createDataFrame(genre_counts) \
        .orderBy('count', ascending=False)
    
    return genre_counts_df

# Get genre counts
genre_counts = count_songs_per_genre(df_raw)

# Show results
print("Number of songs per genre:")
genre_counts.show(100, truncate=False)

# Plot top 20 genres
plt.figure(figsize=(15, 8))
top_20_genres = genre_counts.limit(80).toPandas()
plt.bar(top_20_genres['genre'], top_20_genres['count'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Genre')
plt.ylabel('Number of Songs')
plt.title('Top 20 Genres by Number of Songs')
plt.tight_layout()
plt.show()

# %%
