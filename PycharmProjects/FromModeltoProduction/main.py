import pandas as pd
import pickle
from fetch_data import *
from visualize_control import *
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

#Usage
url = "https://archive-api.open-meteo.com/v1/archive"
latitude = 48.7823
longitude = 9.177
start_date = "2024-01-01"
two_days_ago = datetime.today() - timedelta(days=2)
end_date = two_days_ago.strftime('%Y-%m-%d')

csv_file = "weather_data.csv"

df = fetch_and_process_weather_data(url, latitude, longitude, start_date, end_date)

if df is not None:
	# Save the data to a CSV file
	save_weather_data_to_csv(df, csv_file)

# Read csv file
df = pd.read_csv(csv_file)

# Detect Anomalies and save to Pickle File
def detect_anomalies(df, feature_columns,
					 contamination=0.01,
					 random_state=42,
					 output_pickle_file="isolation_forest_model.pkl"):
	"""
	    Detect anomalies in the given DataFrame using the Isolation Forest algorithm.

	    Parameters:
	        df (pd.DataFrame): The input DataFrame containing the data.
	        feature_columns (list): The list of column names to use as features for the model.
	        contamination (float): The proportion of anomalies in the data.
	        random_state (int): Random seed for reproducibility.

	    Returns:
	        pd.DataFrame: A DataFrame containing anomalies and their counts per day.
	        pd.Series: Anomaly counts grouped by date.
	    """
	# Extract Features for Model
	X = df[feature_columns]

	# Create Isolation Forest Model
	isolation_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
	df["anomaly_score"] = isolation_forest.fit_predict(X)  # -1 for anomalies, 1 for normal

	# Group for date and show accumulation
	anomalies = df[df["anomaly_score"] == -1].copy()
	anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"], errors="coerce")
	anomalies.loc[:, "date"] = anomalies["timestamp"].dt.date  # Extract the Date
	anomaly_counts_per_day = anomalies.groupby("date").size()

	#Integrate statistical measures
	silhouette = silhouette_score(X, isolation_forest.predict(X))
	# Silhouette Score -1 (incorrect clustering) to +1 (well-clustered), with 0 indicating overlapping clusters

	print(f"Silhouette Score: {silhouette:.4f}")

	# Save results to a pickle file
	results = {
		"anomalies": anomalies,
		"anomaly_counts_per_day": anomaly_counts_per_day,
	}

	with open(output_pickle_file, "wb") as file:
		pickle.dump(results, file)

	print(f"Results saved to {output_pickle_file}")
	return anomalies, anomaly_counts_per_day

# Example usage
feature_columns = ["temperature_2m_scaled", "relative_humidity_2m_scaled", "sound_volume"]
anomalies, anomaly_counts_per_day = detect_anomalies(df, feature_columns)


# Plot boxplots and dataframes to see an overview of the data
#plot_dataframe(df)
#boxplot_scaled_variable()
#plot_normal_and_anomalies(df)
#plot_anomalies(df, anomaly_counts_per_day)
#plot_anomaly_score_distribution(df)




# Ich nutze jetzt schon Flask und präsentiere meinen Code da
# Testen mit Beispielen von ChatGPT, ob mein Model richtig klassifiziert
# Etwas verbessern
# In die Cloud laden