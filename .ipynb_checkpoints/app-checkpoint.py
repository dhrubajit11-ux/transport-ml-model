import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# =========================================================
# === Placeholder Data and Model Generation for Testing ===
# =========================================================
# This section creates dummy files so the Flask app can run
# without external dependencies. In a real-world scenario,
# you would have these files pre-trained and saved.

MODEL_PATH = "transport_model.pkl"
DATA_PATH = "data1.csv"

# Required features for the model
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

# Class mapping
class_map = {
    0: "Bus",
    1: "Car",
    2: "Still",
    3: "Train",
    4: "Walking"
}
reverse_class_map = {v: k for k, v in class_map.items()}

# Generate a dummy model and data if files don't exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    print("Generating dummy model and data files for demonstration...")
    
    # Create dummy data
    data = np.random.rand(100, len(required_keys))
    # Create a simple target column where y = 0 if first feature is low, 1 otherwise
    targets = (data[:, 1] > 0.5).astype(int)
    
    df = pd.DataFrame(data, columns=required_keys)
    df['target'] = targets
    df.to_csv(DATA_PATH, index=False)
    
    # Create and train a dummy model
    dummy_model = LogisticRegression()
    dummy_model.fit(df[required_keys], df['target'])
    
    # Save the dummy model
    joblib.dump(dummy_model, MODEL_PATH)
    print("Dummy files generated successfully!")


# =========================================================
# === Flask Application Start ===
# =========================================================
app = Flask(__name__)

# Load model
model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}. Prediction endpoint will be disabled.")

# ================================
# Pre-calculate accuracy once at startup
# ================================
real_accuracy = None
if model:
    try:
        df = pd.read_csv(DATA_PATH)
        print("CSV Columns:", df.columns.tolist())

        # Detect target column automatically
        possible_targets = ["target", "activity", "label", "Transport_Mode"]
        target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
        print("Detected target column:", target_col)

        # Fill missing features with 0
        for col in required_keys:
            if col not in df.columns:
                df[col] = 0

        # Prepare data
        X_test = df[required_keys].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_test = df[target_col]

        # If target column is text, map to numeric
        if y_test.dtype == object or y_test.dtype == 'str':
            y_test = y_test.map(reverse_class_map)
        
        # Ensure y_test has no NaN values after mapping
        y_test = y_test.fillna(-1) # Use a placeholder value for missing labels

        # Predict
        y_pred = model.predict(X_test)
        
        # Filter out rows where the target could not be mapped (y_test == -1)
        valid_indices = y_test != -1
        y_test_valid = y_test[valid_indices]
        y_pred_valid = y_pred[valid_indices]
        
        # Calculate accuracy only on valid data
        real_accuracy_value = accuracy_score(y_test_valid, y_pred_valid) * 100
        real_accuracy = round(real_accuracy_value, 2)
        print(f"Real Accuracy: {real_accuracy}%")
        print("Note: If the accuracy is 100%, it might mean the test data is the same as the training data.")

    except Exception as e:
        print("Error calculating accuracy:", e)
        real_accuracy = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Transport ML Model API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the model file path and name."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data provided"}), 400

        # Validate input
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": "Missing input fields", "missing_keys": missing_keys}), 400

        # Prepare features for single prediction
        features = np.array([[data[key] for key in required_keys]])
        pred_numeric = model.predict(features)[0]
        pred_label = class_map.get(pred_numeric, "Unknown")

        # Format accuracy
        formatted_accuracy = f"{real_accuracy}%" if real_accuracy is not None else "N/A"

        return jsonify({
            "accuracy": formatted_accuracy,
            "prediction": pred_label
        })

    except Exception as e:
        # Provide a more specific error message
        return jsonify({"error": f"Prediction error: {str(e)}. Please check your input data format."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
