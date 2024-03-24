import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

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


def create_and_fit_model(df_aggregated, daily_fourier_order=15):
  """Initializes, configures, and fits the Prophet model."""
  m = Prophet(changepoint_prior_scale=1, seasonality_prior_scale=100)
  m.add_seasonality(name='daily', period=1, fourier_order=daily_fourier_order)
  m.add_seasonality(name='hourly', period=1/24, fourier_order=20)
  m.add_seasonality(name='weekly', period=7, fourier_order=10)
  #m.add_regressor('hour')
  #m.add_regressor('day_of_week')
  m.add_regressor('is_weekend')
  m.fit(df_aggregated)
  return m

def make_and_plot_forecast(model, future_periods=300):
  """Generates forecasts and displays the plot with informative labels."""
  future = model.make_future_dataframe(periods=300, freq='H')
  future['hour'] = future['ds'].dt.hour
  future['day_of_week'] = future['ds'].dt.dayofweek
  future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
  forecast = model.predict(future)
  forecast['yhat'] 
  fig = model.plot(forecast)
  plt.legend(labels=['Actual', 'Trend', 'Daily Seasonality', 'Hourly Seasonality', 'Weekly Seasonality', 'Forecast'])
  plt.title('Prophet Forecast with Adjusted Daily Seasonality')
  plt.xlabel('Date')
  plt.ylabel('Total Pedestrian Count')
  plt.show()
  fig = model.plot_components(forecast)
  plt.show()

if __name__ == '__main__':
  df_aggregated = load_and_preprocess_data(data_path)
  model = create_and_fit_model(df_aggregated, daily_fourier_order=20)
  make_and_plot_forecast(model)
