import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.offline as py

data_path = 'october.csv'

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
    m.add_seasonality(name='hourly', period=1/24, fourier_order=8)
    m.add_seasonality(name='weekly', period=7, fourier_order=8)
    m.add_regressor('hour')
    m.add_regressor('day_of_week')
    m.add_regressor('is_weekend')
    m.fit(df_train)
    return m

def make_and_plot_forecast(model, df_test):
    """Generates forecasts and displays the plot with informative labels."""
    future = model.make_future_dataframe(periods=len(df_test)*2, freq='900S', include_history=False)
    future['hour'] = future['ds'].dt.hour
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    forecast = model.predict(future)

    # 2. Accessing Changepoints
    changepoints_df = model.changepoints

    # Print the first 5 rows of the changepoints DataFrame
    print("Changepoints:")
    print(changepoints_df.head())
    # Set negative predicted values to zero
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    residuals = df_test['y'] - forecast['yhat']

    # Print summary statistics of the residuals
    print("\nResiduals Summary:")
    print(residuals.describe())

    fig = model.plot(forecast)
    plt.scatter(df_test['ds'], df_test['y'], color='r', label='Test Data')
    plt.legend(labels=['Train Data', 'Prediction', 'Error margin', 'Test Data'])
    plt.title('October Prophet Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Pedestrian Count')
    plt.show()
    fig2 = model.plot_components(forecast)
    plt.show()


   
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
    mse = mean_squared_error(y_true, y_pred[0:len(y_true)])
    mae = mean_absolute_error(y_true, y_pred[0:len(y_true)])
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

        # Calculate median, mean, and mode
    median_pedestrians = df_aggregated['y'].median()
    mean_pedestrians = df_aggregated['y'].mean()
    
    print(f"Median number of pedestrians: {median_pedestrians}")
    print(f"Mean number of pedestrians: {mean_pedestrians}")
    
    # Plot histogram
    #plt.hist(df_aggregated['y'], bins=10, color='skyblue', edgecolor='black')
    #plt.title('Histogram of Aggregated Pedestrians')
    #plt.xlabel('Number of Pedestrians')
    #plt.ylabel('Frequency')
    #plt.grid(axis='y', alpha=0.75)
    #plt.show()