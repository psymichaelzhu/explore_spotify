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


#%%
def examine_dataset(df,representative_check=True):
    print("number of observations:",df.count())
    df.printSchema()
    df.show(5,truncate=False)
    #representative check
    if representative_check:
        df.filter(F.col("artist").contains("Oasis")).show(truncate=False)
        df.filter(F.col("artist").contains("Coldplay")).show(truncate=False)
        df.filter(F.col("artist").contains("David Bowie")).show(truncate=False)

df = spark.read.csv('/home/mikezhu/music/data/spotify_dataset.csv', header=True)
examine_dataset(df) 


#%% trend plot
artist_list = ["Coldplay","David Bowie","Oasis","Radiohead"]
df_trending = df.filter(F.col("artist").isin(artist_list))
# Combine data into a single file
df_trending.coalesce(1).write.mode("overwrite").csv("/home/mikezhu/music/results/EDA_trending.csv", header=True)

#%% trend plot
artist_list = ["AURORA","Taylor Swift","Avril Lavigne","Lady Gaga"]
df_trending = df.filter(F.col("artist").isin(artist_list))
# Combine data into a single file
df_trending.coalesce(1).write.mode("overwrite").csv("/home/mikezhu/music/results/EDA_trending2.csv", header=True)



#%%



# %%

def analyze_feature_distributions(df, feature_cols, artist_list=None):
    """
    Analyze and visualize the distribution of musical features
    
    Args:
        df: Spark DataFrame with song data
        feature_cols: List of feature columns to analyze
        artist_list: Optional list of artists to filter by
    """
    # Filter by artists if specified
    if artist_list:
        df = df.filter(F.col("artist").isin(artist_list))
    
    # Convert to pandas for easier plotting and standardize features
    features_pd = df.select(
        "artist",
        *[F.col(col).cast("float").alias(col) for col in feature_cols]
    ).toPandas()
    
    # Standardize each feature column
    for feature in feature_cols:
        mean = features_pd[feature].mean()
        std = features_pd[feature].std()
        features_pd[feature] = (features_pd[feature] - mean) / std
    
    # Create distribution plots
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        
        # Histogram
        ax.hist(features_pd[feature].dropna(), bins=30, alpha=0.6, color='skyblue')
        
        # Add mean and median lines
        mean_val = features_pd[feature].mean()
        median_val = features_pd[feature].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        
        # Add labels and title
        ax.set_title(f'Distribution of Standardized {feature}')
        ax.set_xlabel(f'Standardized {feature}')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Print summary statistics
        print(f"\nSummary statistics for standardized {feature}:")
        print(features_pd[feature].describe())
    
    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Create box plots by artist if artist_list is provided
    if artist_list:
        plt.figure(figsize=(15, 8))
        for feature in feature_cols:
            plt.figure(figsize=(12, 6))
            features_pd.boxplot(column=feature, by='artist')
            plt.title(f'Standardized {feature} by Artist')
            plt.ylabel(f'Standardized {feature}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Example usage:
feature_cols = [
    'key',
    'tempo', 
    #'time_signature',
    'mode',
    'acousticness', 
    'instrumentalness',  
    'speechiness',
    'energy',
    'valence',
    'danceability',
    'loudness',
    'duration_ms'
]

# Analyze all data
print("Overall feature distributions:")
analyze_feature_distributions(df, feature_cols)

# Analyze specific artists
#artist_list = ["Coldplay", "David Bowie", "Oasis", "  Radiohead"]
#print(f"\nFeature distributions for selected artists: {artist_list}")
#analyze_feature_distributions(df, feature_cols, artist_list)

# %%

def remove_outliers(df, feature_cols):
    """
    Remove outliers from specified feature columns using the IQR method
    
    Args:
        df: Spark DataFrame
        feature_cols: List of feature column names to process
        
    Returns:
        Spark DataFrame with outliers removed
    """
    df_clean = df
    
    for col in feature_cols:
        # Calculate quartiles and IQR
        quantiles = df_clean.select(
            F.expr(f'percentile_approx({col}, array(0.25, 0.75))').alias('quantiles')
        ).collect()[0]['quantiles']
        
        Q1 = quantiles[0]
        Q3 = quantiles[1]
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        df_clean = df_clean.filter(
            (F.col(col) >= lower_bound) & 
            (F.col(col) <= upper_bound)
        )
        
        # Print statistics
        print(f"\nOutlier removal statistics for {col}:")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")
        print(f"Records removed: {df.count() - df_clean.count()}")
    
    print(f"\nFinal record count: {df_clean.count()} (Original: {df.count()})")
    return df_clean

# 使用示例:
df_clean = remove_outliers(df, feature_cols)
analyze_feature_distributions(df_clean, feature_cols)

# %%
