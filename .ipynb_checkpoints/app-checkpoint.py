import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "transport_model.pkl")

print("=== DEBUG INFO ===")
print("NumPy version in production:", np.__version__)
print("Current Working Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir(os.getcwd()))
print("Absolute Model Path:", MODEL_PATH)
print("===================")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: transport_model.pkl not found!")
    model = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Transport ML Model API is running!"})


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not trained or not found. Please check the server logs."}), 500

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

        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({
                "error": "Missing input fields",
                "missing_keys": missing_keys
            }), 400

        features = np.array([[data[key] for key in required_keys]])

        prediction = model.predict(features)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
