


# %% innovation levels over time: release date
def plot_innovation_levels_over_time(cluster_data, artist_name=None, sample_size=0.1, seed=None, 
                                   bins=[0, 1, 2, 3, 4], distance_type='historical'):
    """
    Plot the number of songs in different innovation level ranges over time and return the data
    
    Args:
        cluster_data: DataFrame with cluster predictions and features
        artist_name: Optional artist name to filter data
        sample_size: Fraction of data to sample
        seed: Random seed for sampling
        bins: Innovation level ranges (default: [0, 1, 2, 3, 4])
        distance_type: Type of distance to calculate ('historical' or 'internal')
        
    Returns:
        DataFrame containing innovation counts by date and range
    """
    # Extract data
    num_clusters, colors, cluster_data = exact_to_pd(cluster_data, artist_name, sample_size, seed)
    
    # Convert year to datetime and filter for 1920-2020
    # Randomly assign dates within each year
    # Check if release_date only contains year
    if cluster_data['release_date'].str.len().max() == 4:
        # If only year, add random day within that year
        cluster_data['release_date'] = pd.to_datetime(cluster_data['release_date'].astype(str)) + \
            pd.to_timedelta(np.random.randint(0, 365, size=len(cluster_data)), unit='D')
    else:
        # Otherwise just convert to datetime
        cluster_data['release_date'] = pd.to_datetime(cluster_data['release_date'])
        
    cluster_data = cluster_data[
        (cluster_data['release_date'] >= '1980-01-01') & 
        (cluster_data['release_date'] <= '1984-12-31')
    ]
    
    # Sort data by release date
    cluster_data = cluster_data.sort_values('release_date')
    
    # Process date by release_date
    dates = cluster_data['release_date'].unique()
    historical_centroid = None
    
    # Store innovation levels for each date
    innovation_by_date = {date: [] for date in dates}
    
    for date in dates:
        date_data = cluster_data[cluster_data['release_date'] == date]
        if not date_data.empty:
            current_points = np.array([
                [float(features[0]), float(features[1])] 
                for features in date_data['features']
            ])
            current_centroid = current_points.mean(axis=0)
            
            if distance_type == 'historical':
                if historical_centroid is not None:
                    # Calculate distances to historical centroid
                    distances = np.sqrt(
                        np.sum((current_points - historical_centroid) ** 2, axis=1)
                    )
                    innovation_by_date[date].extend(distances)
                
                # Update historical centroid
                if historical_centroid is None:
                    historical_centroid = current_centroid
                else:
                    historical_data = cluster_data[cluster_data['release_date'] <= date]
                    historical_points = np.array([
                        [float(features[0]), float(features[1])] 
                        for features in historical_data['features']
                    ])
                    historical_centroid = historical_points.mean(axis=0)
            
            else:  # internal distance
                # Calculate distances to current date's centroid
                distances = np.sqrt(
                    np.sum((current_points - current_centroid) ** 2, axis=1)
                )
                innovation_by_date[date].extend(distances)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Calculate counts for each bin range
    bin_ranges = list(zip(bins[:-1], bins[1:]))
    counts_by_range = {f"{low:.1f}-{high:.1f}": [] for low, high in bin_ranges}
    
    for date in dates:
        date_innovations = innovation_by_date[date]
        if date_innovations:
            # Count songs in each range
            for (low, high) in bin_ranges:
                count = sum((low <= x < high) for x in date_innovations)
                counts_by_range[f"{low:.1f}-{high:.1f}"].append(count)
        else:
            # Add 0 if no songs on this date
            for range_key in counts_by_range:
                counts_by_range[range_key].append(0)
    
    # Plot lines for each range
    colors = plt.cm.viridis(np.linspace(0, 1, len(bin_ranges)))
    for (range_key, counts), color in zip(counts_by_range.items(), colors):
        plt.plot(dates, counts, '-', label=f'Innovation {range_key}', 
                color=color, linewidth=1, alpha=0.8)
    
    plt.xlabel('Release Date')
    plt.ylabel('Number of Songs')
    title = f"Distribution of {'Historical' if distance_type == 'historical' else 'Internal'} Innovation Levels Over Time"
    if artist_name:
        title = f"{artist_name}: {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Create and return DataFrame with results
    results_df = pd.DataFrame(counts_by_range, index=dates)
    results_df.index.name = 'Release Date'
    
    # Add raw innovation values
    results_df['raw_innovation_values'] = [innovation_by_date[date] for date in dates]
    
    return results_df

# 使用历史距离
'''results_df = plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.01,bins=[0,100])#1,1.5,2,2.5,
plot_innovation_levels_over_time(cluster_results, distance_type='historical',sample_size=0.01,bins=[0,2,100])

# 使用年距离
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.01,bins=[0,100])
plot_innovation_levels_over_time(cluster_results, distance_type='internal',sample_size=0.01,bins=[0,2,100])'''


# %% Check release date distribution: we can't use the release date as a index
# Count how many release dates are January 1st
jan_first_count = df.filter(F.date_format('release_date', 'MM-dd') == '01-01').count()
total_count = df.count()

print(f"\nRelease Date Analysis:")
print(f"Number of songs released on January 1st: {jan_first_count}")
print(f"Total number of songs: {total_count}")
print(f"Percentage of January 1st releases: {(jan_first_count/total_count)*100:.2f}%")

# Count songs where release_date length is 4 (YYYY format)
year_format_count = df.filter(F.length('release_date') == 4).count()

print("\nYear Format (YYYY) Analysis:")
print(f"Number of songs with 4-digit release date: {year_format_count}")
print(f"Total number of songs: {total_count}")
print(f"Percentage of 4-digit release dates: {(year_format_count/total_count)*100:.2f}%")

# %%
