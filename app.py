import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test.csv")

# Define class map
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load test data for real-world accuracy
try:
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Assuming last column is 'target'
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Calculate real-world accuracy
    y_pred = model.predict(X_test)
    real_accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Real-world accuracy: {real_accuracy:.2f}%")

except Exception as e:
    print(f"Error loading test data: {e}")
    X_test, y_test, real_accuracy = None, None, None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Transport ML Model API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Required feature keys
        required_keys = [
            "time",
            "android.sensor.accelerometer#mean",
            "android.sensor.accelerometer#min",
            "android.sensor.accelerometer#max",
            "android.sensor.accelerometer#std",
            "android.sensor.gyroscope#mean",
            "android.sensor.gyroscope#min",
            "android.sensor.gyroscope#max",
            "android.sensor.gyroscope#std",
            "sound#mean",
            "sound#min",
            "sound#max",
            "sound#std"
        ]

        # Check missing features
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        # Prepare input for prediction
        features = np.array([[data[key] for key in required_keys]])

        # Make prediction
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        # Return real-world accuracy and predicted label
        return jsonify({
            "accuracy": round(real_accuracy, 2) if real_accuracy is not None else "N/A",
            "prediction": pred_label
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
