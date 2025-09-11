import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import numpy as np
import logging
import os
from sklearn.metrics import accuracy_score

# Initialize Flask
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables
model = None
le = None
feature_names = None


# ============ TRAIN MODEL FUNCTION ============ #
def train_model():
    global model, le, feature_names
    logging.info("🚀 Starting model training process...")

    try:
        # Check if data1.csv exists
        logging.info(f"Looking for CSV in: {os.getcwd()}")
        logging.info(f"Does data1.csv exist? {os.path.exists('data1.csv')}")

        df = pd.read_csv('data1.csv')
        logging.info("✅ CSV loaded successfully.")

        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        feature_names = X.columns.tolist()

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        logging.info("🎯 Model training complete and ready for predictions.")
        return True

    except FileNotFoundError:
        logging.error("❌ data1.csv not found.")
        return False
    except Exception as e:
        logging.error(f"🔥 An error occurred during training: {e}")
        return False


# ============ ROUTES ============ #
@app.route('/')
def home():
    return "🚀 Transport Mode Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained. Please check the server logs."}), 500

    try:
        # Get JSON data
        data = request.get_json(force=True)

        # Convert JSON to numpy array
        input_features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)

        # Predict
        prediction_encoded = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]
        confidence = prediction_proba[prediction_encoded]

        # Decode label
        predicted_label = le.inverse_transform([prediction_encoded])[0]

        return jsonify({
            "results": [
                {
                    "Prediction": predicted_label,
                    "Accuracy": f"{confidence * 100:.2f}%"
                }
            ]
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature in JSON payload: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500


# ============ START FLASK APP ============ #
if __name__ == '__main__':
    if train_model():
        port = int(os.environ.get("PORT", 5000))  # Required for Render deployment
        app.run(host='0.0.0.0', port=port, debug=False)
