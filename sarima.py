import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from tqdm import tqdm 

def train_or_load_sarima(train, order, seasonal_order, toTrain=False, filename=None):
    if not filename.endswith('.joblib'):
        filename += '.joblib'
    if not toTrain and filename is not None and os.path.exists(filename):
        # Load the model from disk
        with open(filename, 'rb') as f:
            sarima_model = pickle.load(f)
        print(f"Model loaded from '{filename}'.")
    else:
        # Train a new SARIMA model
        print("Training a new SARIMA model, please wait ...")
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        sarima_model = model.fit(disp=False)
        
        if filename is not None:
            if not filename.endswith('.joblib'):
                filename += '.joblib'
            # Save the model to disk
            with open(filename, 'wb') as f:
                pickle.dump(sarima_model, f)
            print(f"Model saved as '{filename}'.")
        
    return sarima_model

# Load and preprocess data
data_path = input("Please enter data path: ")
df = pd.read_csv(data_path, parse_dates=['UTC Datetime', 'Local Datetime'])
df_aggregated = df.groupby(['UTC Datetime', 'Local Datetime']).sum().reset_index()
df_aggregated = df_aggregated.rename(columns={'UTC Datetime': 'ds', 'Pedestrian': 'y'})
df_aggregated.set_index('ds', inplace=True)
df_aggregated.fillna(0, inplace=True)  # Handle missing values if any

# Ensure numeric columns only
df_aggregated = df_aggregated.select_dtypes(include=['float64', 'int64'])

# Set frequency to 15 minutes
df_aggregated.index.freq = '15T'

# Split data into train and test sets
train_proportion = 0.5  # 60% of data for training
train_size = int(len(df_aggregated) * train_proportion)
train, test = df_aggregated.iloc[:train_size], df_aggregated.iloc[train_size:]

# Define SARIMA model parameters
order = (1, 0, 1)  # ARIMA parameters (p, d, q)
seasonal_order = (1, 0, 1, 96)  # SARIMA parameters (P, D, Q, S)

# Define paths for saving and loading the model
model_save_path = input("Please enter the path to save/load the model: ")

# Ask user if they want to train a new model or use the saved one
train_new_model = input("Do you want to train a new SARIMA model? (yes/no): ").lower().strip() == 'yes'

sarima_model = train_or_load_sarima(train['y'], order, seasonal_order,  train_new_model, filename=model_save_path)

# Forecast for the test set
forecast = sarima_model.get_forecast(steps=len(test))
forecast_values = forecast.predicted_mean
forecast_values = forecast_values.clip(lower=0)
# Get the forecast index for the test set
forecast_index = test.index

# Evaluate the model
mse = mean_squared_error(test['y'], forecast_values)
mae = mean_absolute_error(test['y'], forecast_values)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Calculate median, mode, and mean of number of pedestrians
median_pedestrians = df_aggregated['y'].median()
mean_pedestrians = df_aggregated['y'].mean()

print(f"Median number of pedestrians: {median_pedestrians}")
print(f"Mean number of pedestrians: {mean_pedestrians}")

# Visualize the forecast for the test set
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['y'], label='Training data')
plt.plot(test.index, test['y'], label='Test data')
plt.plot(forecast_index, forecast_values, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Pedestrian Count')
plt.title('SARIMA Forecast')
plt.legend()
plt.show()

# Plot histogram of aggregated number of pedestrians
#plt.hist(df_aggregated['y'], bins=10, color='skyblue', edgecolor='black')
#plt.title('Histogram of Aggregated Number of Pedestrians')
#plt.xlabel('Number of Pedestrians')
#plt.ylabel('Frequency')
#plt.show()

# Visualize the data
#plt.figure(figsize=(10, 6))
#plt.plot(train.index, train['y'], label='Training data')
#plt.plot(test.index, test['y'], label='Test data')
#plt.xlabel('Date')
#plt.ylabel('Pedestrian Count')
#plt.title('15-Minute Pedestrian Count')
#plt.legend()
#plt.show()

# Plot ACF and PACF to determine model parameters
#plt.figure(figsize=(12, 6))
#plot_acf(train['y'], lags=50, ax=plt.subplot(121))
#plot_pacf(train['y'], lags=50, ax=plt.subplot(122))
#plt.show()
