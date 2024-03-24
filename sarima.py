import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load and preprocess data
data_path = 'august.csv'
df = pd.read_csv(data_path, parse_dates=['UTC Datetime', 'Local Datetime'])
df_aggregated = df.groupby(['UTC Datetime', 'Local Datetime']).sum().reset_index()
df_aggregated = df_aggregated.rename(columns={'UTC Datetime': 'ds', 'Pedestrian': 'y'})
df_aggregated.set_index('ds', inplace=True)
df_aggregated.fillna(0, inplace=True)  # Handle missing values if any

# Ensure numeric columns only
df_aggregated = df_aggregated.select_dtypes(include=['float64', 'int64'])

# Resample to hourly frequency
df_aggregated = df_aggregated.resample('H').mean()

# Split data into train and test sets
train_proportion = 0.6  # 60% of data for training
train_size = int(len(df_aggregated) * train_proportion)
train, test = df_aggregated.iloc[:train_size], df_aggregated.iloc[train_size:]

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training data')
plt.plot(test, label='Test data')
plt.xlabel('Date')
plt.ylabel('Pedestrian Count')
plt.title('Hourly Pedestrian Count')
plt.legend()
plt.show()

# Plot ACF and PACF to determine model parameters
plt.figure(figsize=(12, 6))
plot_acf(train, lags=50, ax=plt.subplot(121))
plot_pacf(train, lags=50, ax=plt.subplot(122))
plt.show()

# Define and fit SARIMA model
order = (1, 1, 1)  # ARIMA parameters (p, d, q)
seasonal_order = (1, 0, 1, 24)  # SARIMA parameters (P, D, Q, S)
model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
sarima_model = model.fit(disp=False)

# Forecast for the test set
forecast = sarima_model.get_forecast(steps=len(test))
forecast_values = forecast.predicted_mean

# Get the forecast index for the test set
forecast_index = test.index

# Evaluate the model
mse = mean_squared_error(test, forecast_values)
print(f'Mean Squared Error: {mse}')

# Visualize the forecast for the test set
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training data')
plt.plot(test, label='Test data')
plt.plot(forecast_index, forecast_values, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Pedestrian Count')
plt.title('SARIMA Forecast')
plt.legend()
plt.show()

# Forecast for the next three months (assuming 30 days per month)
forecast_steps = 24 * 30 * 3  # Assuming 3 months ahead
forecast_next_months = sarima_model.get_forecast(steps=forecast_steps)

# Get the forecast values for the next three months
forecast_values_next_months = forecast_next_months.predicted_mean

# Get the forecast index for the next three months starting from the end of the test set
forecast_index_next_months = pd.date_range(start=test.index[-1], periods=forecast_steps, freq='H')

# Visualize the forecast for the next three months
plt.figure(figsize=(10, 6))
plt.plot(df_aggregated.index, df_aggregated['y'], label='Actual Data')
plt.plot(forecast_index_next_months, forecast_values_next_months, label='Forecast Next 3 Months')
plt.xlabel('Date')
plt.ylabel('Pedestrian Count')
plt.title('SARIMA Forecast for Next 3 Months')
plt.legend()
plt.show()
