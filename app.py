import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data1.csv")

# Label mapping
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}
reverse_class_map = {v: k for k, v in class_map.items()}

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Transport Model API Running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": "Missing input fields", "missing_keys": missing_keys}), 400

        # Prepare features for prediction
        features = np.array([[data[key] for key in required_keys]])
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        # ================================
        # Calculate Real Accuracy
        # ================================
        try:
            df = pd.read_csv(DATA_PATH)
            print("CSV Columns:", df.columns.tolist())

            # Detect target column
            possible_targets = ["target", "activity", "label", "Transport_Mode"]
            target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
            print("Using target column:", target_col)

            # Ensure required features exist
            for col in required_keys:
                if col not in df.columns:
                    df[col] = 0  # Fill missing with 0

            # Prepare X_test and y_test
            X_test = df[required_keys].apply(pd.to_numeric, errors='coerce').fillna(0)
            y_test = df[target_col]

            # If target is text, convert to numeric
            if y_test.dtype == object or y_test.dtype == 'str':
                y_test = y_test.map(reverse_class_map)

            # Make predictions
            y_pred = model.predict(X_test)

            # Debugging
            print("First 5 actual labels:", y_test.head().tolist())
            print("First 5 predicted labels:", y_pred[:5].tolist())

            # Calculate accuracy
            real_accuracy = accuracy_score(y_test, y_pred) * 100
            formatted_accuracy = f"{real_accuracy:.2f}%"

        except Exception as e:
            print("Error calculating accuracy:", e)
            formatted_accuracy = "N/A"

        return jsonify({
            "accuracy": formatted_accuracy,
            "prediction": pred_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
