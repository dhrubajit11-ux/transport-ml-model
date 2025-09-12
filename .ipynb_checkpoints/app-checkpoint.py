import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data1.csv")

# Map numeric predictions to human-readable labels
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}

# ===========================
# Load trained model
# ===========================
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ===========================
# Load test data and calculate real-world accuracy
# ===========================
real_accuracy = None
target_column = None

try:
    test_df = pd.read_csv(TEST_DATA_PATH)
    print("Columns in data1.csv:", test_df.columns.tolist())

    # Detect target column automatically
    possible_target_cols = ["target", "activity", "label", "Transport_Mode"]
    for col in possible_target_cols:
        if col in test_df.columns:
            target_column = col
            break

    # If not found, assume last column is target
    if target_column is None:
        target_column = test_df.columns[-1]

    print(f"Detected target column: {target_column}")

    # Split into features and target
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Make predictions on the entire test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    acc_value = accuracy_score(y_test, y_pred) * 100
    real_accuracy = round(acc_value, 2)
    print(f"Real-world accuracy: {real_accuracy}%")

except Exception as e:
    print(f"Error loading test data: {e}")
    real_accuracy = None


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

        # Required features
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

        # Validate input
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        # Prepare features for prediction
        features = np.array([[data[key] for key in required_keys]])

        # Predict label
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        # Format accuracy nicely
        formatted_accuracy = f"{real_accuracy}%" if real_accuracy is not None else "N/A"

        return jsonify({
            "accuracy": formatted_accuracy,
            "prediction": pred_label
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ===========================
# Run Flask app
# ===========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
