import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")

# ==========================
# Debug Info
# ==========================
print("=== DEBUG INFO ===")
print("NumPy version:", np.__version__)
print("Working Directory:", os.getcwd())
print("Files:", os.listdir(os.getcwd()))
print("Model Path:", MODEL_PATH)
print("===================")

# ==========================
# Load model
# ==========================
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: transport_model.pkl not found!")
    model = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

# Final class mapping based on your labels
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}

# ==========================
# Default route
# ==========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Transport ML Model API is running!"})

# ==========================
# Single prediction route
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained or not found. Please check server logs."}), 500

    try:
        data = request.get_json()

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

        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        # Prepare features for prediction
        features = np.array([[data[key] for key in required_keys]])

        # Make prediction
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        return jsonify({
            "accuracy": "N/A",
            "prediction": pred_label
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ==========================
# Evaluation route for accuracy and multiple predictions
# ==========================
@app.route("/evaluate", methods=["POST"])
def evaluate():
    if model is None:
        return jsonify({"error": "Model not trained or not found."}), 500

    try:
        data = request.get_json()

        # Expecting features and labels
        X_test = np.array(data["features"])
        y_test = np.array(data["labels"])

        # Predict on test data
        y_pred = model.predict(X_test)

        # Convert numeric predictions to labels
        y_pred_labels = [class_map.get(pred, "Unknown") for pred in y_pred]

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred) * 100
        acc_str = f"{round(acc, 2)}%"

        return jsonify({
            "accuracy": acc_str,
            "predictions": y_pred_labels
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
