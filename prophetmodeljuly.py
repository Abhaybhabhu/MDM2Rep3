import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.offline as py
import numpy as np

data_path = 'august.csv'

def load_and_preprocess_data(data_path):
    """Loads data from CSV, aggregates counts, and extracts features."""
    df = pd.read_csv(data_path, parse_dates=['UTC Datetime', 'Local Datetime'])
    df_aggregated = df.groupby(['UTC Datetime', 'Local Datetime']).sum().reset_index()
    df_aggregated = df_aggregated.rename(columns={'UTC Datetime': 'ds', 'Pedestrian': 'y'})
    df_aggregated['hour'] = df_aggregated['ds'].dt.hour
    df_aggregated['day_of_week'] = df_aggregated['ds'].dt.dayofweek
    df_aggregated['is_weekend'] = (df_aggregated['ds'].dt.dayofweek >= 5).astype(int)
    return df_aggregated


def create_and_fit_model(df_train, daily_fourier_order=13):
    """Initializes, configures, and fits the Prophet model."""
    m = Prophet(changepoint_prior_scale=0.3, seasonality_prior_scale=0.3)
    m.add_seasonality(name='daily', period=1, fourier_order=daily_fourier_order)
    m.add_seasonality(name='hourly', period=1 / 24, fourier_order=8)
    m.add_seasonality(name='weekly', period=7, fourier_order=8)
    m.add_regressor('hour')
    m.add_regressor('day_of_week')
    m.add_regressor('is_weekend')
    m.fit(df_train)
    return m


def make_and_plot_forecast(model, df_test):
    """Generates forecasts and displays the plot with informative labels."""
    future = model.make_future_dataframe(periods=len(df_test), freq='900S', include_history=False)
    future['hour'] = future['ds'].dt.hour
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    forecast = model.predict(future)


    for i in range(len(forecast)):
        if forecast['yhat'][i] < 0:
            forecast['yhat'][i] = 0

    # Set up figure size
    plt.figure(figsize=(12, 6))  # Adjust width and height as needed

    # Plot prediction and actual July data
    plt.plot(df_test['ds'], forecast['yhat'], label='Prediction', color='blue')
    plt.scatter(df_test['ds'], df_test['y'], color='red', label='Actual July Data', s=10)
    plt.legend()
    plt.title('July Prophet Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Pedestrian Count')
    plt.show()



    return forecast



if __name__ == '__main__':
    # Load and preprocess July data
    july_data_path = 'july.csv'
    df_july = load_and_preprocess_data(july_data_path)
    print(len(df_july))

    # Load and preprocess August data
    august_data_path = 'august.csv'
    df_august = load_and_preprocess_data(august_data_path)
    print(len(df_august))

    # Train the model on August data
    model = create_and_fit_model(df_august)

    # Use the trained model to make predictions on July data
    forecast_july = make_and_plot_forecast(model, df_july)

    # Evaluate the model on July data
    y_pred_july = forecast_july['yhat']
    print(len(y_pred_july))

    y_true_july = df_july['y']
    print(len(y_true_july))
    mse_july = mean_squared_error(y_true_july, y_pred_july)
    mae_july = mean_absolute_error(y_true_july, y_pred_july)
    print(f"Mean Squared Error (July): {mse_july}")
    print(f"Mean Absolute Error (July): {mae_july}")

    # Calculate median, mean, and mode for July data
    median_pedestrians_july = df_july['y'].median()
    mean_pedestrians_july = df_july['y'].mean()
    mean_pedestrians_julypred = y_pred_july.mean()

    print(f"Median number of pedestrians (July): {median_pedestrians_july}")
    print(f"Mean number of pedestrians (July): {mean_pedestrians_july}")
    print(f"Mean number of pedestrians predicted (July): {mean_pedestrians_julypred}")
    # Plot histogram
    # plt.hist(df_aggregated['y'], bins=10, color='skyblue', edgecolor='black')
    # plt.title('Histogram of Aggregated Pedestrians')
    # plt.xlabel('Number of Pedestrians')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()