# %% Load data and packages
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.fpm import FPGrowth

# Initialize Spark session
spark = SparkSession \
        .builder \
        .appName("genre_analysis-fp growth") \
        .getOrCreate()

# %% Load and process data
df = spark.read.csv('/home/mikezhu/data/spotify_data.csv', header=True)

# Convert string representation of list to array
genre_df = df.select(F.from_json(
    F.col('genres'), 
    ArrayType(StringType())
).alias('items'))

# %% Perform FP-Growth analysis
fp = FPGrowth( 
    minSupport=0.005, 
    minConfidence=0.4,
    
)
fpm = fp.fit(genre_df)

# %% FP-Growth results
# Show frequent itemsets sorted by frequency
print("Frequent Genre Sets:")
fpm.freqItemsets.sort("freq", ascending=False).show()

# Show association rules with lift calculation
print("\nGenre Association Rules:")
fpm.associationRules.sort("antecedent", "consequent").show()

# %% Visualization
# preparations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

# Convert Spark DataFrame to Pandas for visualization
freq_genres = fpm.freqItemsets.toPandas()
rules = fpm.associationRules.toPandas()

# %% 1. Confidence vs Lift Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(rules['confidence'], rules['lift'], alpha=0.5)
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Confidence vs Lift for Genre Associations')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)  # Reference line for lift=1
plt.tight_layout()
plt.show()
# %% 2. Top Frequent Genres Bar Plot
def plot_top_n_genres(freq_genres_df, n_genres, top_n=20):
    """
    Plot top N most frequent genre combinations of specified size
    
    Parameters:
    -----------
    freq_genres_df : pandas.DataFrame
        DataFrame containing frequency analysis results
    n_genres : int
        Number of genres in combination (1 for single, 2 for pairs, 3 for triplets)
    top_n : int
        Number of top combinations to plot
    """
    plt.figure(figsize=(12, 6))
    
    # Filter for combinations of specified size
    filtered_genres = freq_genres_df[freq_genres_df['items'].apply(len) == n_genres]
    
    # Create readable labels
    if n_genres == 1:
        filtered_genres['label'] = filtered_genres['items'].apply(lambda x: x[0])
        title = f'Top {top_n} Most Frequent Single Genres'
        xlabel = 'Genre'
    else:
        filtered_genres['label'] = filtered_genres['items'].apply(lambda x: ' , '.join(x))
        title = f'Top {top_n} Most Frequent Genre {"Pairs" if n_genres == 2 else "Triplets"}'
        xlabel = f'Genre {"Pairs" if n_genres == 2 else "Triplets"}'
    
    # Get top N combinations
    top_genres = filtered_genres.nlargest(top_n, 'freq')
    
    # Create plot
    plt.bar(top_genres['label'], top_genres['freq'])
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot for different combination sizes
for n in [1, 2, 3]:
    plot_top_n_genres(freq_genres, n)

# %% 3. Association Rules Network
def plot_single_item_rules_network(rules, min_confidence=0.5, top_rules=50):
    """
    优化的单个元素关联规则网络图，添加动态节点大小、边透明度和图例
    
    Parameters:
    -----------
    rules : pandas.DataFrame
        DataFrame containing association rules
    min_confidence : float
        Minimum confidence threshold for filtering rules
    top_rules : int
        Number of top rules to display
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 过滤规则，只保留 antecedent 和 consequent 为单元素的规则
    filtered_rules = rules[
        (rules['confidence'] >= min_confidence) & 
        (rules['antecedent'].apply(len) == 1) &
        (rules['consequent'].apply(len) == 1)
    ].nlargest(top_rules, 'confidence')
    
    # 添加边和权重
    for _, rule in filtered_rules.iterrows():
        antecedent = rule['antecedent'][0]  # 单元素取第一个
        consequent = rule['consequent'][0] # 单元素取第一个
        G.add_edge(antecedent, consequent, 
                   weight=rule['confidence'], 
                   lift=rule['lift'])
    
    # 动态调整节点大小（根据度数）
    node_sizes = [500 + 1000 * (G.in_degree(node) + G.out_degree(node)) for node in G.nodes()]
    
    # 动态调整边透明度（根据置信度）
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    edge_alphas = [min(0.3 + weight, 1) for weight in weights]
    
    # 布局优化
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.8)  # 调整布局参数
    
    # 先绘制节点
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color='skyblue',
        node_size=node_sizes,
        alpha=0.9
    )
    
    # 绘制边和箭头（在节点之后）
    for edge, alpha in zip(G.edges(), edge_alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge],
            edge_color=[G[edge[0]][edge[1]]['weight']],
            edge_cmap=plt.cm.viridis_r,  # 使用反转的鸢尾花配色
            edge_vmin=min(weights),
            edge_vmax=max(weights),
            width=2,
            alpha=alpha,
            arrows=True,
            arrowsize=15,
            node_size=node_sizes  # 添加节点大小参数以正确计算箭头位置
        )
    
    # 绘制标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.3)
    )
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis_r,  # 使用反转的鸢尾花配色
        norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Confidence', rotation=270, labelpad=15)
    
    # 添加标题和图例说明
    plt.title('Optimized Single-Item Genre Association Network', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 绘制优化后的网络图
plot_single_item_rules_network(rules, min_confidence=0.40)

# %%
