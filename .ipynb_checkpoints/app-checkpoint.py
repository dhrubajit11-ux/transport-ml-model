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
TEST_DATA_PATH = os.path.join(BASE_DIR, "test.csv")

# Map numeric predictions to human-readable labels
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Required features (must match the model's training features)
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

# ROUTES
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Transport ML Model API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Validate input
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        # Prepare features for prediction
        features = np.array([[data[key] for key in required_keys]])

        # Predict label for input
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        # ===========================
        # Calculate real-world accuracy dynamically
        # ===========================
        formatted_accuracy = "N/A"
        try:
            test_df = pd.read_csv(TEST_DATA_PATH)
            print("Columns in CSV:", test_df.columns.tolist())

            # Detect target column
            possible_target_cols = ["target", "activity", "label", "Transport_Mode"]
            target_column = next((col for col in possible_target_cols if col in test_df.columns), None)
            if target_column is None:
                target_column = test_df.columns[-1]
            print("Using target column:", target_column)

            # Check missing features
            missing_features = [col for col in required_keys if col not in test_df.columns]
            if missing_features:
                print("Missing features in CSV:", missing_features)
                # Fill missing columns with zeros
                for col in missing_features:
                    test_df[col] = 0

            # Align features and convert to float safely
            X_test = test_df[required_keys].copy()
            X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            y_test = test_df[target_column]

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            real_accuracy = accuracy_score(y_test, y_pred) * 100
            formatted_accuracy = f"{real_accuracy:.2f}%"
            print("Calculated real-world accuracy:", formatted_accuracy)

        except Exception as e:
            print("Accuracy calculation error:", e)
            formatted_accuracy = "N/A"

        # Return JSON
        return jsonify({
            "accuracy": formatted_accuracy,
            "prediction": pred_label
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
