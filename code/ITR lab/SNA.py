# %% time series analysis functions

import seaborn as sns
def prepare_data(data, year=1970): 
            """
            Parameters
            ----------
            data : pandas.DataFrame 
                The dataframe to prepare, needs to have a datetime index
            year: integer 
                The year separating the training set and the test set (includes the year)

            Returns
            -------
            data_train : pandas.DataFrame
                The training set, formatted for fbprophet.
            data_test :  pandas.Dataframe
                The test set, formatted for fbprophet.
            """
            print(data)
            data_train = data.loc[:str(year - 1),:]
            data_test = data.loc[str(year):,:]
            data_train.reset_index(inplace=True)
            data_test.reset_index(inplace=True)
            print(data_train)
            print(data_test)
            data_train = data_train.rename({'datetime':'ds'}, axis=1)
            data_test = data_test.rename({'datetime':'ds'}, axis=1)
            print(data_train)
            return data_train, data_test
def make_verif(forecast, train_df, test_df):
        """
        Combine forecast and observed data for verification plotting
        
        Args:
            forecast: DataFrame from Prophet forecast
            train_df: Training data DataFrame
            test_df: Test data DataFrame
            
        Returns:
            Combined DataFrame with forecasts and observations
        """
        # Set datetime index for all DataFrames
        forecast.index = pd.to_datetime(forecast.ds)
        train_df.index = pd.to_datetime(train_df.ds)
        test_df.index = pd.to_datetime(test_df.ds)
        
        # Combine training and test data
        full_data = pd.concat([train_df, test_df], axis=0)
        
        # Add observed values to forecast DataFrame
        forecast.loc[:,'y'] = full_data.loc[:,'y']
        
        return forecast

def plot_correlation_analysis(verif, train_df, test_df):
    """
    Plot correlation analysis between forecast and observations using Seaborn jointplots.

    Args:
        verif: Combined DataFrame with forecasts and observations.
        train_df: Training data DataFrame.
        test_df: Test data DataFrame.
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    # Split data into train and test periods
    train_mask = verif.index.isin(train_df.index)
    test_mask = verif.index.isin(test_df.index)
    
    train_data = verif[train_mask]
    test_data = verif[test_mask]
    
    # Create train period jointplot
    g1 = sns.jointplot(
        data=train_data,
        x='yhat',
        y='y',
        kind='reg',
        height=8,
        marginal_kws=dict(bins=30),
        joint_kws={'scatter_kws': dict(alpha=0.5)}
    )
    g1.fig.suptitle('Training Period: Forecast vs Observations', y=1.02)
    r_train = train_data['yhat'].corr(train_data['y'])
    mae_train = np.mean(np.abs(train_data['yhat'] - train_data['y']))
    g1.ax_joint.text(0.05, 0.95, f'R = {r_train:.3f}\nMAE = {mae_train:.3f}',
                   transform=g1.ax_joint.transAxes, verticalalignment='top')
    
    # Create test period jointplot
    g2 = sns.jointplot(
        data=test_data,
        x='yhat',
        y='y',
        kind='reg', 
        height=8,
        marginal_kws=dict(bins=30),
        joint_kws={'scatter_kws': dict(alpha=0.5)}
    )
    g2.fig.suptitle('Test Period: Forecast vs Observations', y=1.02)
    r_test = test_data['yhat'].corr(test_data['y'])
    mae_test = np.mean(np.abs(test_data['yhat'] - test_data['y']))
    g2.ax_joint.text(0.05, 0.95, f'R = {r_test:.3f}\nMAE = {mae_test:.3f}',
                   transform=g2.ax_joint.transAxes, verticalalignment='top')
    plt.tight_layout()
    plt.show()



#%% time series analysis
def time_series_analysis(results_df,cluster_data, artist_name=None, sample_size=0.1, seed=None,
                                   bins=[0, 1, 2, 3, 4], distance_type='historical',
                                   forecast_years=10, cycle_years=40, holidays=None, events=None,
                                   cut_off_year=1970):
    """
    Analyze innovation levels using Prophet with holidays and custom regressors for streaming periods.
    
    Args:
        cluster_data: DataFrame with cluster predictions and features.
        artist_name: Optional artist name to filter data.
        sample_size: Fraction of data to sample.
        seed: Random seed for sampling.
        bins: Innovation level ranges.
        distance_type: Type of distance to calculate ('historical' or 'internal').
        forecast_years: Number of years to forecast.
        cycle_years: Length of cycle in years for custom seasonality.
        holidays: List of dates for holidays or special events.
    """
    from prophet import Prophet
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Step 1: Prepare innovation data
    #results_df
    # Step 2: Define holidays if provided
    holidays_df = None
    if holidays:
        holidays_df = pd.DataFrame({
            'holiday': 'innovation_peak',
            'ds': pd.to_datetime(holidays),
            'lower_window': -2,
            'upper_window': 2,
        })
    
    prophet_results = {}
    bin_ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    
    for range_name in bin_ranges:
        # Step 3: Create DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(results_df.index.astype(str) + '-01-01'),
            'y': results_df[range_name]
        })
        
        # Add regressors for streaming events
        for event_year in events:
            df[f'year_since_{event_year}'] = (df['ds'].dt.year - event_year).clip(lower=0)
            df[f'streaming_{event_year}'] = (df['ds'] >= f'{event_year}-01-01').astype(int)
        
        # Split into train and test sets
        train_df, test_df = prepare_data(df, cut_off_year)
        
        # Step 4: Initialize Prophet model
        model = Prophet(
            mcmc_samples=300, 
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            holidays_prior_scale=10.0,
            n_changepoints=0
        )
        
        # Add custom seasonality
        model.add_seasonality(
            name='long_cycle',
            period=cycle_years * 365.25,
            fourier_order=3,
            prior_scale=0.1
        )
        
        # Add regressors for intercept and slope changes
        for event_year in events:
            model.add_regressor(f'streaming_{event_year}',prior_scale=20)
            model.add_regressor(f'year_since_{event_year}',prior_scale=10)
        
        # Step 5: Fit the model on training data
        model.fit(train_df)
        
        # Step 6: Make predictions on test data and future
        future = model.make_future_dataframe(periods=forecast_years, freq='Y')
        for event_year in events:
            future[f'year_since_{event_year}'] = (future['ds'].dt.year - event_year).clip(lower=0)
            future[f'streaming_{event_year}'] = (future['ds'] >= f'{event_year}-01-01').astype(int)
        
        forecast = model.predict(future)
        test_forecast = model.predict(test_df)
        
        # Step 7: Store results
        prophet_results[range_name] = {
            'model': model,
            'forecast': forecast,
            'train': train_df,
            'test': test_df,
            'test_forecast': test_forecast
        }
        
        # Step 8: Visualization
       
        # In the time_series_analysis function, replace the verification section with:
        verif = make_verif(forecast, train_df, test_df)
        
        # Generate correlation plots
        plot_correlation_analysis(verif, train_df, test_df)

        raise Exception('stop here')
        # Plot components
        model.plot_components(forecast)
        plt.suptitle(f'Analysis for Innovation Range {range_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Plot forecast with regressors
        plt.figure(figsize=(12, 6))
        fig = model.plot(forecast)
        ax = fig.gca()
        plt.title(f'Innovation Level Forecast for Range {range_name}\n(Including Regressor Effects)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Plot regressor effects
        regressors = [f'streaming_{event_year}' for event_year in events] + [f'year_since_{event_year}' for event_year in events]
        regressor_effects = []
        
        # Safely access regressor coefficients from model parameters
        for name in regressors:
            try:
                # Convert params['beta'] to a dictionary if it's a Series/DataFrame
                beta_params = dict(zip(model.extra_regressors.keys(), 
                                     model.params['beta'].flatten()))
                effect = beta_params.get(name, 0)
                regressor_effects.append(effect)
            except Exception as e:
                print(f"Warning: Could not get effect for regressor {name}: {e}")
                regressor_effects.append(0)
        
        plt.figure(figsize=(10, 5))
        plt.bar(regressors, regressor_effects, color='skyblue')
        plt.title('Regressor Effects on Innovation Levels', fontsize=14)
        plt.ylabel('Effect Size')
        plt.xlabel('Regressors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return prophet_results

# Example usage:
prophet_results = time_series_analysis(
    results_df, 
    cluster_results, 
    distance_type='historical',#historical, internal
    sample_size=0.001,
    bins=[0,2,100],#0,0.5,1, 1.5, 2, 2.5, 100
    forecast_years=0,
    cycle_years=40,#40
    holidays=['1966-01-01', '1993-01-01'],
    events=[2000],
    cut_off_year=1970
)

# %%
#自动：不同参数：长周期；傅立叶阶数；强度
#检验
#拼图

#更大比例；不同距离；seed

