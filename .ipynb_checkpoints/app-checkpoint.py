import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import numpy as np
import logging
from sklearn.metrics import accuracy_score

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables for the model and label encoder
model = None
le = None
feature_names = None

def train_model():
    global model, le, feature_names
    logging.info("Starting model training process...")
    
    try:
        # Load data
        df = pd.read_csv('data1.csv')
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        feature_names = X.columns.tolist()
        
        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Train model
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
    try:
        data = request.get_json(force=True)

        # ----- CASE 1: Single Prediction -----
        if isinstance(data, dict):
            input_features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)
            
            prediction_encoded = model.predict(input_features)[0]
            prediction_proba = model.predict_proba(input_features)[0]
            
            confidence = prediction_proba[prediction_encoded]
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
        
        # ----- CASE 2: Batch Accuracy Check -----
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            
            if 'target' not in df.columns:
                return jsonify({"error": "Missing 'target' column in input data"}), 400
            
            X = df.drop(columns=['target'])
            y = df['target']
            
            predictions_encoded = model.predict(X)
            predictions = le.inverse_transform(predictions_encoded)
            
            correctness = (predictions == y).astype(int)
            overall_accuracy = accuracy_score(y, predictions)
            
            results = pd.DataFrame({
                "Actual": y,
                "Predicted": predictions,
                "Correct(1)/Incorrect(0)": correctness
            })
            
            return jsonify({
                "overall_accuracy": f"{overall_accuracy * 100:.2f}%",
                "results": results.to_dict(orient="records")
            })
        
        else:
            return jsonify({"error": "Invalid input format. Send a JSON object or list."}), 400

    except KeyError as e:
        return jsonify({"error": f"Missing feature in JSON payload: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    if train_model():
        app.run(host='0.0.0.0', port=5000, debug=False)
