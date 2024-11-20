
def plot_cluster_radar(predictions_df, original_df, artist_name, feature_cols, num_clusters):
    """
    Create a radar plot to compare cluster centroids for a specific artist with standardized features
    """
    # Merge predictions with original features
    df_merged = predictions_df.withColumn("id", F.monotonically_increasing_id()) \
            .join(original_df.withColumn("id", F.monotonically_increasing_id())
                 .withColumnRenamed("features", "raw_features"), on="id", how="inner")
    
    # Calculate global means and standard deviations for standardization
    global_stats = df_merged.select([
        *[F.avg(F.expr(f"raw_features[{i}]")).alias(f"{col}_mean") for i, col in enumerate(feature_cols)],
        *[F.stddev(F.expr(f"raw_features[{i}]")).alias(f"{col}_std") for i, col in enumerate(feature_cols)]
    ]).collect()[0]
    
    # Filter for specific artist
    artist_data = df_merged.filter(F.col("artist") == artist_name)
    
    # Calculate cluster means and standardize them
    cluster_means = artist_data.groupBy("prediction").agg(*[
        F.avg(F.expr(f"raw_features[{i}]")).alias(col)
        for i, col in enumerate(feature_cols)
    ])
    
    # Convert to pandas for plotting
    cluster_means_pd = cluster_means.toPandas()
    
    # Ensure all clusters are represented
    all_clusters = pd.DataFrame({'prediction': range(num_clusters)})
    cluster_means_pd = pd.merge(all_clusters, cluster_means_pd, on='prediction', how='left')
    
    # Standardize the means using global statistics
    for i, col in enumerate(feature_cols):
        mean_val = global_stats[f"{col}_mean"]
        std_val = global_stats[f"{col}_std"]
        if std_val > 0:  # Avoid division by zero
            cluster_means_pd[col] = (cluster_means_pd[col] - mean_val) / std_val
    
    # Set up the angles for
    angles = np.linspace(0, 2*np.pi, len(feature_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the circle
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Define colors
    colors = plt.cm.Dark2(np.linspace(0, 1, num_clusters))
    
    # Plot data for each cluster
    for cluster in range(num_clusters):
        values = cluster_means_pd[cluster_means_pd['prediction'] == cluster][feature_cols].values
        if len(values) > 0 and not np.isnan(values).all():
            values = values[0]
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}', 
                   color=colors[cluster])
            ax.fill(angles, values, alpha=0.25, color=colors[cluster])
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=8)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(f"Cluster Characteristics for {artist_name}")
    plt.tight_layout()
    plt.show()
