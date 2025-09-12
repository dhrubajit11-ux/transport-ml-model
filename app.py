import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data1.csv")

class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}
reverse_class_map = {v: k for k, v in class_map.items()}

try:
    model = joblib.load(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

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


real_accuracy = None
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"CSV Columns: {df.columns.tolist()}")

    possible_targets = ["target", "activity", "label", "Transport_Mode"]
    target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
    logging.info(f"Detected target column: {target_col}")

    for col in required_keys:
        if col not in df.columns:
            logging.warning(f"Missing column in CSV, filling with 0: {col}")
            df[col] = 0

    X_test = df[required_keys].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_test = df[target_col]

    if y_test.dtype == object or y_test.dtype == "str":
        y_test = y_test.map(reverse_class_map)

    y_pred = model.predict(X_test)
    real_accuracy_value = accuracy_score(y_test, y_pred) * 100
    real_accuracy = round(real_accuracy_value, 2)
    logging.info(f"Real-world model accuracy: {real_accuracy}%")

except Exception as e:
    logging.error(f"Error calculating accuracy: {e}")
    real_accuracy = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Transport ML Model API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)

        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        features = np.array([[data[key] for key in required_keys]])

        pred_numeric = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        confidence = probabilities[pred_numeric]

        pred_label = class_map.get(pred_numeric, "Unknown")

        response = {
            "prediction": pred_label,
            "accuracy": f"{confidence * 100:.2f}%"
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
