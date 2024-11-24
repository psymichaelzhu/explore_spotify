#%%
# Import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.feature import StandardScaler as StandardScalerRDD
from pyspark.mllib.linalg.distributed import RowMatrix
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

spark = SparkSession \
        .builder \
        .appName("preprocess_eda") \
        .getOrCreate()

df_raw = spark.read.csv('/home/mikezhu/music/data/60W/tracks.csv', header=True)

#%% examine dataset by artist and id
def examine_dataset(df,representative_check=True):
    print("number of observations:",df.count())
    df.printSchema()
    df.show(5,truncate=False)
    #representative check
    if representative_check:
        df.filter(F.col("artists").contains("Oasis")).show(truncate=False)
        df.filter(F.col("artists").contains("Coldplay")).show(truncate=False)


examine_dataset(df_raw) 




# %% Check the data type of release_date column
print("\nRelease Date Column Data Type:")
print(df_raw.select('release_date').dtypes)

# Show a few example release dates
print("\nSample Release Dates:")
df_raw.select('release_date').show(5)


# Count how many release dates are January 1st
jan_first_count = df_raw.filter(F.date_format('release_date', 'MM-dd') == '01-01').count()
total_count = df_raw.count()

print(f"\nRelease Date Analysis:")
print(f"Number of songs released on January 1st: {jan_first_count}")
print(f"Total number of songs: {total_count}")
print(f"Percentage of January 1st releases: {(jan_first_count/total_count)*100:.2f}%")

# Count songs where release_date length is 4 (YYYY format)
year_format_count = df_raw.filter(F.length('release_date') == 4).count()

print("\nYear Format (YYYY) Analysis:")
print(f"Number of songs with 4-digit release date: {year_format_count}")
print(f"Total number of songs: {total_count}")
print(f"Percentage of 4-digit release dates: {(year_format_count/total_count)*100:.2f}%")


#%%
'''
high quality dataset 
586672 observations
variables:
 |-- id: string (nullable = true)
 |-- name: string (nullable = true)
 |-- popularity: string (nullable = true)
 |-- duration_ms: string (nullable = true)
 |-- explicit: string (nullable = true)
 |-- artists: string (nullable = true)
 |-- id_artists: string (nullable = true)
 |-- release_date: string (nullable = true)
 |-- danceability: string (nullable = true)
 |-- energy: string (nullable = true)
 |-- key: string (nullable = true)
 |-- loudness: string (nullable = true)
 |-- mode: string (nullable = true)
 |-- speechiness: string (nullable = true)
 |-- acousticness: string (nullable = true)
 |-- instrumentalness: string (nullable = true)
 |-- liveness: string (nullable = true)
 |-- valence: string (nullable = true)
 |-- tempo: string (nullable = true)
 |-- time_signature: string (nullable = true)
'''
#%% preprocessing
#artists column: ['Rage Against Th.']; some songs have multiple artists ['Ariana Grande', 'Iggy Azalea']

df=df_raw
# Extract id and artists columns; Convert string representation of list to actual list and explode to multiple rows; Merge artist as a new column
df_artists = df.select(
    "id",
    F.regexp_replace(F.regexp_replace("artists", "\\[|\\]", ""), "\"", "").alias("artists")
)\
.select(
    "id",
    F.split("artists", "', '").alias("artists_array")
)\
.select(
    "id",
    F.expr("transform(artists_array, x -> regexp_replace(x, \"'\", ''))").alias("artists_clean")
)

# Explode the array to create multiple rows for songs with multiple artists
df_artists_exploded = df_artists.select("id", F.explode("artists_clean").alias("artist"))

#Merge the exploded artists data back to the original dataframe
df = df.join(df_artists_exploded, "id", "left")


#transform integer columns to integer
integer_columns = [
    "duration_ms", 
    "key", 
    "loudness",
    "time_signature", 
    "mode",
    "popularity"
    ]
for column in integer_columns:
    df = df.withColumn(column, F.col(column).cast("integer"))

#transform boolean columns to boolean
boolean_columns = [
    "explicit",
    ]
for column in boolean_columns:
    df = df.withColumn(column, F.col(column).cast("boolean"))

#transform numeric columns to float
double_columns = [
    "speechiness", 
    "acousticness", 
    "instrumentalness",
    "tempo",
    "danceability", 
    "energy", 
    "valence", 
    "liveness",
    ]

for column in double_columns:
    df = df.withColumn(column, F.col(column).cast("double"))

#把release_date转换为datetime
df = df.withColumn("release_date", F.to_date(F.col("release_date")))

#删除artists和id_artists列
df = df.drop("artists","id_artists")



#%% summary: artists & tracks
df.groupBy("artist").count().orderBy(F.col("count").desc()).show(10,truncate=False)
df.groupBy("id").count().orderBy(F.col("count").desc()).show(10,truncate=False)

print("number of artists:",df.select("artist").distinct().count())
print("number of songs:",df.select("id").distinct().count())
examine_dataset(df,representative_check=False)

#%%
#save the processed dataframe
df.coalesce(1).write.mode("overwrite").csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True)

# %%
#artist songs over time
def get_artist_songs(dataset,artist_name,metrics = ['danceability', 'energy', 'valence']):
    """
    Get all songs by a given artist, sorted by release date.
    Also plot various track features over time.
    
    Args:
        artist_name: Name of the artist to analyze
    """
    # Get songs by artist sorted by date
    artist_songs = dataset.filter(F.col("artist") == artist_name)\
                        .orderBy(F.col("release_date"))
    
    # Convert to pandas for plotting
    songs_pd = artist_songs.toPandas()
    print(songs_pd.head(5))
    
    # Plot metrics over time
    
    plt.figure(figsize=(12,8))
    for metric in metrics:
        #string to numeric
        songs_pd[metric] = pd.to_numeric(songs_pd[metric], errors='coerce')

        # Scale the metric to 0-1 range
        songs_pd[metric + "_scaled"] = (songs_pd[metric] - songs_pd[metric].min()) / (songs_pd[metric].max() - songs_pd[metric].min())
        plt.plot(songs_pd['release_date'], songs_pd[metric+"_scaled"], 'o-', label=metric, alpha=0.6)
    
    plt.title(f'Song Metrics Over Time - {artist_name}')
    plt.xlabel('Release Date') 
    plt.ylabel('Metric Value (Scaled 0-1)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Display songs
    artist_songs.show(truncate=False)



get_artist_songs(df,"Coldplay")
get_artist_songs(df,"Taylor Swift")


#%%
#筛选并且保存coldplay和David Bowie的所有数据
df_trending = df.filter(F.col("artist").isin(["Coldplay","David Bowie"]))
# Combine data into a single file
df_trending.coalesce(1).write.mode("overwrite").csv("/home/mikezhu/music/results/trending.csv", header=True)



#%%
def plot_compare_artists_metrics_over_time(df, artists, metrics=['popularity','valence']):
    """
    Plot metrics over time for multiple artists on the same timeline
    """
    plt.figure(figsize=(15, 5*len(metrics)))
    
    # Get all songs for all artists at once and convert to pandas
    artist_songs = df.filter(F.col("artist").isin(artists))\
                    .orderBy(F.col("release_date"))
    songs_pd = artist_songs.toPandas()
    
    # Convert release_date to datetime
    songs_pd['release_date'] = pd.to_datetime(songs_pd['release_date'])
    
    # Create subplots for each metric
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, idx)
        
        # Convert metric to numeric
        songs_pd[metric] = pd.to_numeric(songs_pd[metric], errors='coerce')
            
        # Plot each artist's data
        for artist in artists:
            artist_data = songs_pd[songs_pd['artist'] == artist].sort_values('release_date')
            plt.plot(artist_data['release_date'], 
                    artist_data[metric], 
                    'o-', label=artist, alpha=0.6)
        
        plt.title(f'{metric.capitalize()} Over Time')
        plt.xlabel('Release Date')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage:
artists_to_compare = ["Taylor Swift", "Coldplay","David Bowie"]
plot_compare_artists_metrics_over_time(df, artists_to_compare)


