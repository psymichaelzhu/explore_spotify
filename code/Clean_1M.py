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
import numpy as np
import re
import matplotlib.pyplot as plt

# 确保 SparkSession 正确初始化
if SparkSession._instantiatedSession is None:
    spark = SparkSession \
            .builder \
            .appName("dr_cluster") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
else:
    spark = SparkSession.builder.getOrCreate()

# Read movie data (after changing column names)
df_raw = spark.read.csv('/home/mikezhu/data/1M_Movie.csv', header=True)
print(df_raw.columns)

# %%
#title和director组合的重复
df_raw.groupBy("title","director").count().orderBy(F.col("count").desc()).show()
#大于1的有几个
df_raw.groupBy("title","director").count().filter(F.col("count") > 1).count()
#把重复的删掉
df_raw_unique = df_raw.dropDuplicates(["title","director"])
df_raw_unique.count()

#title和director组合的重复
df_raw_unique.groupBy("title","director").count().orderBy(F.col("count").desc()).show()
#大于1的有几个
df_raw_unique.groupBy("title","director").count().filter(F.col("count") > 1).count()

df_raw=df_raw_unique

# %% examine null
def examine_null(df):
    """
    Analyze null values in a dataframe and return null counts and percentages
    
    Args:
        df: Spark dataframe to analyze
        
    Returns:
        total_rows: Total number of rows in dataframe
        null_analysis: Pandas dataframe with null counts and percentages
    """
    # Calculate null counts and percentages for each column
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    total_rows = df.count()

    # Convert to pandas for better display
    null_analysis = null_counts.toPandas().transpose()
    null_analysis.columns = ['null_count']
    null_analysis['null_percentage'] = (null_analysis['null_count'] / total_rows * 100).round(2)

    # Sort by null percentage in descending order
    null_analysis = null_analysis.sort_values('null_percentage', ascending=False)
    
    # Display results 
    print(f"Total rows: {total_rows}")
    print("\nNull value distribution:")
    print(null_analysis)

examine_null(df_raw)


# %%
def examine_zero(df):
    """
    Analyze zero values in a dataframe and return zero counts and percentages
    
    Args:
        df: Spark dataframe to analyze
        
    Returns:
        total_rows: Total number of rows in dataframe
        zero_analysis: Pandas dataframe with zero counts and percentages
    """
    # Calculate zero counts and percentages for each column
    zero_counts = df.select([F.count(F.when(F.col(c) == 0, c)).alias(c) for c in df.columns])
    total_rows = df.count()

    # Convert to pandas for better display
    zero_analysis = zero_counts.toPandas().transpose()
    zero_analysis.columns = ['zero_count']
    zero_analysis['zero_percentage'] = (zero_analysis['zero_count'] / total_rows * 100).round(2)

    # Sort by zero percentage in descending order
    zero_analysis = zero_analysis.sort_values('zero_percentage', ascending=False)
    
    # Display results 
    print(f"Total rows: {total_rows}")
    print("\nZero value distribution:")
    print(zero_analysis)

examine_zero(df_raw)

# %% wrong director name
df_raw.filter(F.col("director") == 0).show()


df_raw.select("director").distinct().show()# %%
df_raw.select("director").distinct().count()
#189112
# %% 清除非人名director
from nameparser import HumanName
import pandas as pd

name_list = df_raw.select("director").distinct().toPandas()
print("original:\n"+str(name_list.head(20)))

# 定义检查是否为人名的函数
def is_person_name(name):
    name = str(name)
    # 如果包含了标点符号直接返回false
    if re.search(r'[^\w\s]', name):
        return False
    # 如果包含数字直接返回false
    if re.search(r'\d', name):
        return False
    # No Language直接返回false
    if name == "No Language":
        return False
    parsed_name = HumanName(name)
    # 检查名字和姓氏字段是否非空
    return bool(parsed_name.first) and bool(parsed_name.last)

# 应用函数过滤
name_list['is_person'] = name_list['director'].apply(is_person_name)
name_list_cleaned = name_list[name_list['is_person'] == True].drop(columns=['is_person'])

print("cleaned:\n"+str(name_list_cleaned.head(20)))
# 统计比例
print("ratio:\n"+str(name_list_cleaned.shape[0]/name_list.shape[0]))

# %%
# 使用 createDataFrame 将 pandas DataFrame 转换为 Spark DataFrame
name_list_cleaned_spark = spark.createDataFrame(name_list_cleaned)

# 修改过滤方式
df_filtered = df_raw.join(name_list_cleaned_spark, "director", "inner")
filtered_count = df_filtered.count()

#%%
#显示原始数据中id为879120的行
df_raw.filter(F.col("id") == 1089605).show()
#https://www.imdb.com/title/tt15647058/ 对不上号


#这个数据集非常地混乱，很多地方是错乱的；又或者和真实不符合，还有很多重复
#还有runtime是0的


#%%
df_filtered.show()

# %%
# 
name_list_cleaned_sorted = name_list_cleaned.sort_values(by='director', ascending=False)
print(name_list_cleaned_sorted.head(20))

# 统计每个导演的作品数量
director_counts = df_raw.groupBy("director").count()

# 转换为pandas dataframe并与cleaned name list合并
director_counts_pd = director_counts.toPandas()
name_list_with_counts = pd.merge(name_list_cleaned, director_counts_pd, on='director', how='inner')

# 按作品数量降序排序
name_list_with_counts_sorted = name_list_with_counts.sort_values(by='count', ascending=False)

print("\nDirectors sorted by number of works:")
print(name_list_with_counts_sorted.head(20))

# %%
# 导演作品 histgram
plt.hist(name_list_with_counts_sorted['count'], bins=range(1, 100, 2), edgecolor='black')
plt.xlabel('Number of Works')
plt.ylabel('Frequency')
plt.title('Histogram of Director Works')
plt.show()


# %%    
revenue_table = df_raw.groupBy("revenue").count()
revenue_table.show()
# 倒序
revenue_table.orderBy(F.col("count").desc()).show()
# %%
df_raw.filter(F.col("title") == "Pulp Fiction").show()
#revenue:213,900,000 #unit:dollar | worldwide | box office
df_raw.filter(F.col("title") == "Bird Box").show()
#budget:198,000,000 #unit:dollar | worldwide 
df_raw.filter(F.col("title") == "The Assassin").show()
#The Assassin: 15,000,000 #unit:dollar | worldwide | box office
df_raw.filter(F.col("title") == "The Killing Jar").show()
#The Killing Jar: $2,339 box office worldwide | budget $400,000
#%%
revenue_threshold = 2
df_raw.filter(F.col("revenue") == revenue_threshold).show()


# %%
