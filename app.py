from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "transport_model.pkl"
DATA_PATH = "data1.csv"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", str(e))
    model = None

# Class mapping
class_map = {
    0: "Still",
    1: "Walk",
    2: "Run",
    3: "Bike",
    4: "Train"
}
reverse_class_map = {v: k for k, v in class_map.items()}

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
    "android.sensor.gyroscope#std"
]

# ==========================
# Calculate Real Accuracy
# ==========================
def calculate_real_accuracy():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"{DATA_PATH} does not exist!")

        df = pd.read_csv(DATA_PATH)
        print("\n--- Debugging Accuracy Calculation ---")
        print("CSV Columns:", df.columns.tolist())

        # Detect target column automatically
        possible_targets = ["target", "activity", "label", "Transport_Mode"]
        target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
        print("Detected target column:", target_col)

        # Validate data
        if df.empty:
            raise ValueError("CSV file is empty!")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV!")

        # Fill missing feature columns with 0
        for col in required_keys:
            if col not in df.columns:
                print(f"Warning: Missing feature column -> {col}")
                df[col] = 0

        # Prepare features and target
        X_test = df[required_keys].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_test = df[target_col]

        print("First 5 rows of target column:", y_test.head().tolist())

        if y_test.isnull().all():
            raise ValueError("Target column has only null values!")

        # If target labels are text, map them to numeric
        if y_test.dtype == object or y_test.dtype == 'str':
            print("Mapping text labels to numeric using reverse_class_map")
            y_test = y_test.map(reverse_class_map)

        print("First 5 mapped target values:", y_test.head().tolist())

        # Predict using the model
        y_pred = model.predict(X_test)
        print("First 5 predictions:", y_pred[:5].tolist())

        # Calculate accuracy
        real_accuracy_value = accuracy_score(y_test, y_pred) * 100
        real_accuracy = round(real_accuracy_value, 2)
        print(f"Calculated Real Accuracy: {real_accuracy}%")

        return real_accuracy

    except Exception as e:
        print("Error calculating real accuracy:", str(e))
        return None

# ==========================
# Prediction Endpoint
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded!"}), 500

    try:
        data = request.get_json()
        print("\nIncoming request data:", data)

        # Ensure all required features are present
        input_features = []
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing feature: {key}"}), 400
            input_features.append(float(data[key]))

        # Convert to numpy array
        input_array = np.array([input_features])

        # Make prediction
        prediction = model.predict(input_array)[0]
        prediction_label = class_map.get(prediction, "Unknown")

        # Calculate real accuracy
        real_accuracy = calculate_real_accuracy()

        return jsonify({
            "accuracy": f"{real_accuracy:.2f}%" if real_accuracy is not None else "N/A",
            "prediction": prediction_label
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

# ==========================
# Run Flask
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
