import numpy as np
from sklearn.preprocessing import StandardScaler
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

# Setup Open-Meteo API client
def setup_openmeteo_client():
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	return openmeteo_requests.Client(session = retry_session)

# Function to fetch and process data
def fetch_and_process_weather_data(url, latitude, longitude, start_date, end_date):
	# Initialize the API client
	openmeteo = setup_openmeteo_client()

	# Define the API endpoint and parameters
	params = {
		"url": url,
		"latitude": latitude,
		"longitude": longitude,
		"start_date": start_date,
		"end_date": end_date,
		"hourly": ["temperature_2m", "relative_humidity_2m"],
	}

	# Fetch the data
	responses = openmeteo.weather_api(url, params=params)
	response = responses[0]

	# Process hourly data
	hourly = response.Hourly()
	temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

	# Reshape the data for scaling
	hourly_temperature_2m_reshape = hourly.Variables(0).ValuesAsNumpy().reshape(-1, 1)
	hourly_relative_humidity_2m_reshape = hourly.Variables(1).ValuesAsNumpy().reshape(-1, 1)

	# Initialize scalers and scale the data
	scaler_temp = StandardScaler()
	scaler_relative_humidity = StandardScaler()
	scaled_temperature_2m = scaler_temp.fit_transform(hourly_temperature_2m_reshape)
	scaled_relative_humidity = scaler_relative_humidity.fit_transform(hourly_relative_humidity_2m_reshape)

	# Generate white noise
	num_samples = len(temperature_2m)
	white_noise = np.random.normal(0, 1, num_samples)

	# Create the DataFrame
	hourly_data = {
		"timestamp": pd.date_range(
			start=pd.to_datetime(hourly.Time(), unit="s"),
			end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
			freq=pd.Timedelta(seconds=hourly.Interval()),
			inclusive="left"
		),
		"temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "temperature_2m_scaled": scaled_temperature_2m.flatten(),
        "relative_humidity_2m_scaled": scaled_relative_humidity.flatten(),
        "sound_volume": white_noise
	}

	return pd.DataFrame(data=hourly_data)

# Function to save data to a CSV file
def save_weather_data_to_csv(df, csv_file):
	"""
    Save weather data to a CSV file, adding only new data.

    Args:
        df (pd.DataFrame): DataFrame containing the new data.
        csv_file (str): Path to the CSV file.
    """
	if os.path.exists(csv_file):
		# Read the existing data
		existing_data = pd.read_csv(csv_file, parse_dates=["timestamp"])

		# Remove duplicates by comparing timestamps
		new_data = df[~df["timestamp"].isin(existing_data["timestamp"])]
	else:
		# If the file doesn't exist, all data is new
		new_data = df

	if not new_data.empty:
		# Append new data to the CSV file
		new_data.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))
		print(f"{len(new_data)} new rows added to {csv_file}.")
	else:
		print("No new data to add.")