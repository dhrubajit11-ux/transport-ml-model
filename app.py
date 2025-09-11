import os
import logging
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variables
model = None
le = None
feature_names = None


# ----------------------------
# Model training function
# ----------------------------
def train_model():
    global model, le, feature_names
    logging.info("🚀 Starting model training process...")

    try:
        # Current directory
        current_dir = os.getcwd()
        logging.info(f"Current working directory: {current_dir}")

        # Check CSV existence
        csv_path = "data1.csv"
        if not os.path.exists(csv_path):
            logging.error("❌ data1.csv NOT FOUND! Training cannot continue.")
            return False

        # Load CSV
        df = pd.read_csv(csv_path)
        logging.info(f"✅ CSV loaded successfully. Shape: {df.shape}")

        # Check for 'target' column
        if 'target' not in df.columns:
            logging.error("❌ 'target' column missing in CSV!")
            return False

        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']

        logging.info(f"Training data features: {list(X.columns)}")
        logging.info(f"Number of rows: {len(df)}")

        feature_names = X.columns.tolist()

        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        logging.info("🎯 Model training complete and ready for predictions.")
        return True

    except Exception as e:
        logging.error(f"🔥 An error occurred during training: {str(e)}", exc_info=True)
        return False


# ----------------------------
# Predict endpoint
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    global model, feature_names, le

    if model is None:
        return jsonify({"error": "Model not trained. Please check the server logs."}), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        logging.info(f"Received data: {data}")

        # Convert input JSON to dataframe
        input_df = pd.DataFrame([data])

        # Ensure all required features are present
        missing_features = [f for f in feature_names if f not in input_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Reorder columns to match training data
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)[0]
        prediction_label = le.inverse_transform([prediction])[0]
        confidence = max(model.predict_proba(input_df)[0]) * 100

        return jsonify({
            "results": [
                {
                    "Prediction": prediction_label,
                    "Accuracy": f"{confidence:.2f}%"
                }
            ]
        })

    except Exception as e:
        logging.error(f"🔥 Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


# ----------------------------
# Root route
# ----------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the ML Model API! Use /predict endpoint to get predictions."})


# ----------------------------
# Main entry point
# ----------------------------
if __name__ == '__main__':
    if train_model():
        port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000
        logging.info(f"Starting server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logging.error("❌ Model training failed. Server will not start.")
