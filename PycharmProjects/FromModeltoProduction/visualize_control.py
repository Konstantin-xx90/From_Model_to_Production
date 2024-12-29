import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

csv_file = "weather_data.csv"
# Read csv file
df = pd.read_csv(csv_file)

# Boxplot Scaled Variables
def boxplot_scaled_variable():
	# List of features to plot
	features = ["temperature_2m_scaled", "relative_humidity_2m_scaled", "sound_volume"]

	# Create boxplot
	plt.figure(figsize=(10, 6))  # Adjust the figure size
	plt.boxplot([df[feature] for feature in features], vert=False, patch_artist=True)

	# Add labels for each boxplot
	plt.yticks(range(1, len(features) + 1), features)  # Align feature names with boxplots
	plt.title("Boxplot of Multiple Features")
	plt.xlabel("Scaled Values")
	plt.grid(axis="x", linestyle="--", alpha=0.7)

	plt.show()

# Function to plot all variables over the period
def plot_dataframe(df):
	# Prepare data for plotting (optional - select specific columns)
	df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
	df_resampled = df.resample('h', on='timestamp').first()

	dates = df_resampled.index
	sound_volume = df_resampled["sound_volume"]
	temperature_2m = df_resampled["temperature_2m"]
	humidity = df_resampled["relative_humidity_2m"]
	temperature_scaled = df_resampled["temperature_2m_scaled"]
	humidity_scaled = df_resampled["relative_humidity_2m_scaled"]

	# Create the plot
	plt.figure(figsize=(12, 6))
	# Plot sound volume (blue line)
	plt.plot(dates, sound_volume, label="Sound Volume", color='blue')
	plt.plot(dates, temperature_2m, label="Unscaled Temperature", color='red')
	plt.plot(dates, temperature_scaled, label="Scaled Temperature", color='red')
	plt.plot(dates, humidity, label="Unscaled Humidity", color='green')
	plt.plot(dates, humidity_scaled, label="Scaled Humidity", color='green')
	# Set labels and title
	plt.xlabel("Date")
	plt.ylabel("Value")
	plt.title("Sound Volume, Scaled Temperature & Humidity (Past 2 Days)")
	# Add legend
	plt.legend()
	# Rotate x-axis labels for better readability with many data points
	plt.xticks(rotation=45)
	# Show the plot
	plt.tight_layout()
	plt.show()

# Plot Anomalies on Scatter and over time
def plot_anomalies(df, anomaly_counts_per_day):
	"""
    Plots anomalies and their daily distribution.
    Args:
        df (pd.DataFrame): The DataFrame containing the data with anomaly scores.
        anomaly_counts_per_day (pd.Series): A Series with the counts of anomalies per day.
    """
	# Scatter Plot of Anomalies
	plt.figure(figsize=(10,6))
	scatter = plt.scatter(
        df["temperature_2m_scaled"],
        df["relative_humidity_2m_scaled"],
        c=df["anomaly_score"],
        cmap="coolwarm",
        s=(df["sound_volume"] - df["sound_volume"].min() + 1) * 10,  # Size depends on Sound Volume
        alpha=0.7,
        label="Daten"
    )
	plt.xlabel("Temperature (scaled)")
	plt.ylabel("Humidity (scaled)")
	plt.title("Isolation Forest - Anomalies (with Sound Volume)")
	plt.colorbar(scatter, ticks=[-1, 0, 1], label="Anomaly Score")
	plt.legend()
	plt.show()

	# Bar Plot of Anomalies per Day
	plt.figure(figsize=(10, 6))
	anomaly_counts_per_day.plot(kind="bar", color="red", alpha=0.7)
	plt.title("Daily Anomalies Tag")
	plt.xlabel("Date")
	plt.ylabel("Number of Anomalies")
	plt.xticks(rotation=45)
	plt.grid(axis="y", linestyle="--", alpha=0.7)
	plt.tight_layout()
	plt.show()

# Compare normal and anomalies for each value
def plot_normal_and_anomalies(df):

	# Separate anomalies and normal points
	anomalies = df[df["anomaly_score"] == -1]
	normal = df[df["anomaly_score"] == 1]

	# Create a figure with 3 subplots (one for each variable)
	fig, axes = plt.subplots(3, 1, figsize=(12, 15))

	# Plot Temperature Distribution in the first subplot
	axes[0].hist(normal["temperature_2m_scaled"], bins=30, alpha=0.5, label="Normal - Temperature", color='red')
	axes[0].hist(anomalies["temperature_2m_scaled"], bins=30, alpha=0.5, label="Anomalies - Temperature", color='black')
	axes[0].set_title("Temperature Distribution (Normal vs Anomalies)")
	axes[0].set_xlabel("Scaled Value")
	axes[0].set_ylabel("Frequency")
	axes[0].legend()

	# Plot Sound Volume Distribution in the second subplot
	axes[1].hist(normal["sound_volume"], bins=30, alpha=0.5, label="Normal - Sound Volume", color='blue')
	axes[1].hist(anomalies["sound_volume"], bins=30, alpha=0.5, label="Anomalies - Sound Volume", color='black')
	axes[1].set_title("Sound Volume Distribution (Normal vs Anomalies)")
	axes[1].set_xlabel("Scaled Value")
	axes[1].set_ylabel("Frequency")
	axes[1].legend()

	# Plot Humidity Distribution in the third subplot
	axes[2].hist(normal["relative_humidity_2m_scaled"], bins=30, alpha=0.5, label="Normal - Humidity", color='green')
	axes[2].hist(anomalies["relative_humidity_2m_scaled"], bins=30, alpha=0.5, label="Anomalies - Humidity", color='black')
	axes[2].set_title("Humidity Distribution (Normal vs Anomalies)")
	axes[2].set_xlabel("Scaled Value")
	axes[2].set_ylabel("Frequency")
	axes[2].legend()

	# Adjust layout for better spacing between subplots
	plt.tight_layout()

	# Show the plot
	plt.show()

# Plot Anomaly Score Distribution
def plot_anomaly_score_distribution(df):
	plt.hist(df[df["anomaly_score"] == 1]["anomaly_score"], bins=50, alpha=0.5, label="Normal")
	plt.hist(df[df["anomaly_score"] == -1]["anomaly_score"], bins=50, alpha=0.5, label="Anomalies")
	plt.title("Anomaly Scores Distribution")
	plt.legend()
	plt.show()

# Inject Anomalies to detect if Outliers are classified correct
def inject_synthetic_anomalies(df):
	# Assuming 'df' is your dataset
	num_anomalies = 10
	anomalies = df.sample(num_anomalies)
	df_with_anomalies = df.copy()

	# Inject synthetic anomalies
	for col in ['temperature_2m_scaled', 'relative_humidity_2m_scaled', "sound_volume"]:
		df_with_anomalies.loc[anomalies.index, col] = np.random.uniform(low=-100, high=100, size=num_anomalies)

	# Use Isolation Forest on the new dataset with anomalies
	X = df_with_anomalies[["temperature_2m_scaled", "relative_humidity_2m_scaled", "sound_volume"]]
	isolation_forest = IsolationForest(contamination=0.01)
	df_with_anomalies["anomaly_score"] = isolation_forest.fit_predict(X)

	# Now, you can analyze the anomaly detection performance on synthetic anomalies

	# Group for date and show accumulation
	anomalies_outlier = df_with_anomalies[df_with_anomalies["anomaly_score"] == -1] # -1 for anomalies
	anomalies_outlier["timestamp"] = pd.to_datetime(anomalies_outlier["timestamp"], errors="coerce")
	anomalies_outlier.loc[:, "date"] = anomalies_outlier["timestamp"].dt.date  # Extract the Date
	anomaly_outlier_counts_per_day = anomalies_outlier.groupby("date").size()

	plot_anomalies(df_with_anomalies, anomaly_outlier_counts_per_day)

