from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# Function to detect anomalies
def detect_anomalies(df, feature_columns, contamination=0.01, random_state=42):
    """
    Detect anomalies in the given DataFrame using the Isolation Forest algorithm.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature_columns (list): The list of column names to use as features for the model.
        contamination (float): The proportion of anomalies in the data.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing anomalies and their counts per day.
    """
    # Extract Features for Model
    X = df[feature_columns]

    # Create Isolation Forest Model
    isolation_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    df["anomaly_score"] = isolation_forest.fit_predict(X)  # -1 for anomalies, 1 for normal

    # Group for date and show accumulation
    anomalies = df[df["anomaly_score"] == -1].copy()
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"], errors="coerce")
    anomalies["date"] = anomalies["timestamp"].dt.date  # Extract the Date
    anomaly_counts_per_day = anomalies.groupby("date").size()

    # Convert date keys to strings
    anomaly_counts_per_day = {str(date): count for date, count in anomaly_counts_per_day.items()}

    # Integrate statistical measures
    silhouette = silhouette_score(X, isolation_forest.predict(X))

    # Save results to a dictionary
    results = {
        "anomalies": anomalies.to_dict(orient="records"),
        "anomaly_counts_per_day": anomaly_counts_per_day,
        "silhouette_score": round(silhouette, 4),
    }

    return results


@app.route('/detect-anomalies', methods=['POST'])
def detect_anomalies_api():
    """
    API endpoint to detect anomalies in the given data.
    """
    try:
        # Parse input JSON
        input_data = request.json
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Load input DataFrame and feature columns
        data = pd.DataFrame(input_data.get("data", []))
        feature_columns = input_data.get("feature_columns", [])
        if data.empty or not feature_columns:
            return jsonify({"error": "Data or feature_columns missing"}), 400

        # Check if timestamp is present
        if "timestamp" not in data.columns:
            return jsonify({"error": "'timestamp' column is required in the data"}), 400

        # Detect anomalies
        results = detect_anomalies(data, feature_columns)

        # Save results to a pickle file (optional)
        with open("results.pkl", "wb") as file:
            pickle.dump(results, file)

        # Return anomalies and daily counts
        return jsonify({
            "anomalies": results["anomalies"],
            "anomaly_counts_per_day": results["anomaly_counts_per_day"],
            "silhouette_score": results["silhouette_score"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)