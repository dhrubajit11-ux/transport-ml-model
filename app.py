import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import numpy as np
import logging
from sklearn.metrics import accuracy_score

# Correct way to initialize Flask app
app = Flask(__name__)  # <-- FIXED HERE
logging.basicConfig(level=logging.INFO)

# Global variables
model = None
le = None
feature_names = None


# Function to train model
def train_model():
    global model, le, feature_names
    logging.info("Starting model training process...")
    
    try:
        # Load dataset
        df = pd.read_csv('data1.csv')
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        feature_names = X.columns.tolist()
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Train the RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)
        
        logging.info("Model training complete. Ready for predictions.")
    except FileNotFoundError:
        logging.error("data1.csv not found. Please ensure the file is in the same directory.")
        return False
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        return False
    
    return True


@app.route('/')
def home():
    return "🚀 Transport Mode Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained. Please check the server logs."}), 500
        
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert JSON to array in correct feature order
        input_features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)
        
        # Make prediction
        prediction_encoded = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]
        
        confidence = prediction_proba[prediction_encoded]
        
        # Decode label
        predicted_label = le.inverse_transform([prediction_encoded])[0]
        
        response_data = {
            "results": [
                {
                    "Prediction": predicted_label,
                    "Accuracy": f"{confidence * 100:.2f}%"
                }
            ]
        }
        
        return jsonify(response_data)
        
    except KeyError as e:
        return jsonify({"error": f"Missing feature in JSON payload: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500


# Correct way to run Flask app
if __name__ == '__main__':  # <-- FIXED HERE
    if train_model():
        app.run(debug=True)  # Use debug=True while developing
