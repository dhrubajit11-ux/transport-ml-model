import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data1.csv")

# Map numeric predictions to labels
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}

# ===========================
# Load ML model
# ===========================
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ===========================
# Load test data to calculate real-world accuracy
# ===========================
real_accuracy = None
target_column = None

try:
    # Load test.csv
    test_df = pd.read_csv(TEST_DATA_PATH)
    print("Columns in test.csv:", test_df.columns.tolist())

    # Automatically detect target column
    # We assume the target column is the last one OR named 'target' or 'activity'
    possible_target_cols = ["target", "activity"]

    # If known columns exist, use them
    for col in possible_target_cols:
        if col in test_df.columns:
            target_column = col
            break

    # If none matched, take the last column as target
    if target_column is None:
        target_column = test_df.columns[-1]

    print(f"Detected target column: {target_column}")

    # Split data into features and labels
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    real_accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Real-world accuracy: {real_accuracy:.2f}%")

except Exception as e:
    print(f"Error loading test data: {e}")
    X_test, y_test, real_accuracy = None, None, None


# ===========================
# ROUTES
# ===========================
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

        # Check if any required field is missing
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        # Prepare input for prediction
        features = np.array([[data[key] for key in required_keys]])

        # Predict
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        # Build response
        response = {
            "accuracy": round(real_accuracy, 2) if real_accuracy is not None else "N/A",
            "prediction": pred_label
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ===========================
# Run the app
# ===========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
