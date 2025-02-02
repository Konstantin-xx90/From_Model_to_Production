from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# Import Anomalie-Detection
from main import detect_anomalies

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
