import pandas as pd
import itertools
import numpy as np
import prophet
from prophet.diagnostics import cross_validation
import prophet.diagnostics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly

data_path = 'october.csv'

param_grid = {  
    'changepoint_prior_scale': [0.02, 0.01, 0.03, 0.04],
    'seasonality_prior_scale': [0.08, 0.1, 0.12, 0.14],
}

def load_and_preprocess_data(data_path):
    """Loads data from CSV, aggregates counts, and extracts features."""
    df = pd.read_csv(data_path, parse_dates=['UTC Datetime', 'Local Datetime'])
    df_aggregated = df.groupby(['UTC Datetime', 'Local Datetime']).sum().reset_index()
    df_aggregated = df_aggregated.rename(columns={'UTC Datetime': 'ds', 'Pedestrian': 'y'})
    df_aggregated['hour'] = df_aggregated['ds'].dt.hour
    df_aggregated['day_of_week'] = df_aggregated['ds'].dt.dayofweek
    df_aggregated['is_weekend'] = (df_aggregated['ds'].dt.dayofweek >= 5).astype(int)
    df_aggregated['reading_week'] = ([0] *96*29 + [1, 1] *96)
    print(df_aggregated.info())
    return df_aggregated

def create_and_fit_model(df_train, daily_fourier_order=15):
    """Initializes, configures, and fits the Prophet model."""

        # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = prophet.Prophet(**params).fit(df_train)  # Fit model with given params
        df_cv = prophet.diagnostics.cross_validation(m, horizon='5 days',initial="10 days", parallel="processes")
        df_p = prophet.diagnostics.performance_metrics(df_cv, rolling_window=1)

        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    print(tuning_results)

    m = prophet.Prophet(changepoint_prior_scale=5, seasonality_prior_scale=5)
    m.add_seasonality(name='daily', period=1, fourier_order=daily_fourier_order)
    m.add_seasonality(name='hourly', period=1/24, fourier_order=20)
    m.add_seasonality(name='weekly', period=7, fourier_order=10)
    m.add_regressor('hour')
    m.add_regressor('day_of_week')
    m.add_regressor('is_weekend')
    #m.add_regressor('reading_week')
    m.fit(df_train)
    return m

def make_and_plot_forecast(model, df_test):
    """Generates forecasts and displays the plot with informative labels."""
    future = model.make_future_dataframe(periods=len(df_test), freq='H', include_history=False)
    future['hour'] = future['ds'].dt.hour
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    #future['reading_week'] = ([0] *96*29 + [1, 1] *96)
    forecast = model.predict(future)
    
    # Set negative predicted values to zero
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    
    fig = model.plot(forecast)
    plt.scatter(df_test['ds'], df_test['y'], color='r', label='Test Data')
    plt.legend(labels=['Trend', 'Daily Seasonality', 'Hourly Seasonality', 'Weekly Seasonality', 'Forecast', 'Test Data'])
    plt.title('Prophet Forecast with Adjusted Daily Seasonality')
    plt.xlabel('Date')
    plt.ylabel('Total Pedestrian Count')
    plt.show()
    plot_plotly(model, forecast)
    return forecast


if __name__ == '__main__':
    df_aggregated = load_and_preprocess_data(data_path)
    
    # Assuming 20% of the data for testing
    test_size = int(0.4 * len(df_aggregated))
    df_train = df_aggregated[:-test_size]
    df_test = df_aggregated[-test_size:]
    
    model = create_and_fit_model(df_train)
    forecast = make_and_plot_forecast(model, df_test)
    
    # Evaluate the model
    y_pred = forecast['yhat']
    y_true = df_test['y']
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

        # Calculate median, mean, and mode
    median_pedestrians = df_aggregated['y'].median()
    mean_pedestrians = df_aggregated['y'].mean()
    
    print(f"Median number of pedestrians: {median_pedestrians}")
    print(f"Mean number of pedestrians: {mean_pedestrians}")
    
    # Plot histogram
    plt.hist(df_aggregated['y'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram of Aggregated Pedestrians')
    plt.xlabel('Number of Pedestrians')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
