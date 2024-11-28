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
df_1 = spark.read.csv('/home/mikezhu/data/spotify_data.csv', header=True).select('track_id', 'album_name', 'genres',  'artist_followers', 'artist_popularity', 'available_markets')
df_2 = spark.read.csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True).withColumnRenamed('id', 'track_id')
df_1.show(5, truncate=False)
df_2.show(5, truncate=False)
# %%
# 将df_1和df_2按照track_id进行合并
df_merged = df_1.join(df_2, on='track_id', how='inner')
df_merged.show(5, truncate=True)
df_merged.count()



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

# Count number of songs per genre (unique songs only)
genre_counts = df_1.select('track_id', 'genres').rdd \
    .flatMap(lambda x: [(extract_genres(x.genres)[0], x.track_id)] if extract_genres(x.genres) else []) \
    .distinct() \
    .map(lambda x: (x[0], 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: Row(genre=x[0], count=x[1]))

# Convert to DataFrame and sort
genre_counts_df = spark.createDataFrame(genre_counts) \
    .orderBy('count', ascending=False)

# Show results
print("Number of unique songs per genre:")
genre_counts_df.show(100, truncate=False)

# Plot top 20 genres
plt.figure(figsize=(15, 8))
top_20_genres = genre_counts_df.limit(50).toPandas()
plt.bar(top_20_genres['genre'], top_20_genres['count'])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Genre')
plt.ylabel('Number of Unique Songs')
plt.title('Top 20 Genres by Number of Unique Songs')
plt.tight_layout()
plt.show()

df_1.count()
# %%
# Count items with empty genres
empty_genres_count = df_1.filter(F.col('genres') == '[]').count()
print(f"Number of items with empty genres: {empty_genres_count}")

# Calculate percentage
total_count = df_1.count()
empty_genres_percentage = (empty_genres_count / total_count) * 100
print(f"Percentage of items with empty genres: {empty_genres_percentage:.2f}%")

# Rock
empty_genres_count = df_1.filter(F.col('genres').contains('rock')).count()
print(f"Number of items with rock genres: {empty_genres_count}")

# Calculate percentage
total_count = df_1.count()
empty_genres_percentage = (empty_genres_count / total_count) * 100
print(f"Percentage of items with rock genres: {empty_genres_percentage:.2f}%")

# Classical
empty_genres_count = df_1.filter(F.col('genres').contains('classical')).count()
print(f"Number of items with classical genres: {empty_genres_count}")

# Calculate percentage
total_count = df_1.count()
empty_genres_percentage = (empty_genres_count / total_count) * 100
print(f"Percentage of items with classical genres: {empty_genres_percentage:.2f}%")


# Pop
empty_genres_count = df_1.filter(F.col('genres').contains('pop')).count()
print(f"Number of items with pop genres: {empty_genres_count}")

# Calculate percentage
total_count = df_1.count()
empty_genres_percentage = (empty_genres_count / total_count) * 100
print(f"Percentage of items with pop genres: {empty_genres_percentage:.2f}%")


# Rap
empty_genres_count = df_1.filter(F.col('genres').contains('rap')).count()
print(f"Number of items with rap genres: {empty_genres_count}")

# Calculate percentage
total_count = df_1.count()
empty_genres_percentage = (empty_genres_count / total_count) * 100
print(f"Percentage of items with rap genres: {empty_genres_percentage:.2f}%")

# %%
def analyze_genre_distribution(df, genres_list):
    """
    Analyze the distribution of music genres. Songs that don't match specified genres are marked as 'Others'
    
    Args:
        df: Spark DataFrame containing genres column
        genres_list: List of genres to analyze
    Returns:
        None (prints statistics and displays pie chart)
    """
    from functools import reduce
    # Initialize counters
    total_count = df.count()
    genre_counts = {}
    
    # Count empty genres
    empty_count = df.filter(F.col('genres') == '[]').count()
    genre_counts['Empty genres'] = empty_count
    
    # Count songs for each specified genre
    for genre in genres_list:
        count = df.filter(F.col('genres').contains(genre)).count()
        genre_counts[genre] = count
    
    # Count songs that don't belong to any of the specified genres and aren't empty
    conditions = [~F.col('genres').contains(genre) for genre in genres_list]
    conditions.append(F.col('genres') != '[]')
    others_count = df.filter(reduce(lambda a, b: a & b, conditions)).count()
    genre_counts['Others'] = others_count
    
    # Print counts and percentages
    print("\nGenre Distribution:")
    for category, count in genre_counts.items():
        percentage = (count / total_count) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(genre_counts.values(), 
            labels=genre_counts.keys(), 
            autopct='%1.1f%%')
    plt.title('Distribution of Music Genres')
    plt.axis('equal')
    plt.show()

# Example usage
genres_to_analyze = ['rock', 'classical', 'pop', 'rap']
analyze_genre_distribution(df_1, genres_to_analyze)



# %% utilities
def find_songs_by_genre(df, target_genre):
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

# Example usage
rock_songs = find_songs_by_genre(df_1, 'rock')

# %%
# Merge rock_songs with df_2
merged_df = rock_songs.join(df_2, on='track_id', how='inner')
print("\nMerged DataFrame:")
merged_df.show()
merged_df.count()

# %%
