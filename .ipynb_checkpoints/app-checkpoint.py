import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import numpy as np
import logging

from sklearn.metrics import accuracy_score


app = Flask(_name_)
logging.basicConfig(level=logging.INFO)

# Global variables for the model and label encoder
model = None
le = None
feature_names = None

def train_model():
    global model, le, feature_names
    logging.info("Starting model training process...")
    
    try:
        df = pd.read_csv('data1.csv')
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        feature_names = X.columns.tolist()
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
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
    
        # Get data from Postman request
        data = request.get_json(force=True)
        
        # Extract features and convert to DataFrame
        # Ensure the order matches the training data
        input_features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)
        
        # Predict the class and the probability of that class
        prediction_encoded = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]
        
        # Get the confidence for the predicted class
        confidence = prediction_proba[prediction_encoded]
        
        # Decode the predicted class label
        predicted_label = le.inverse_transform([prediction_encoded])[0]
        
        # Format the response as a dictionary
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
        # Get JSON data from request
        data = request.get_json()

        # Ensure it's a list of records
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of JSON objects"}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame(data)

        # Ensure 'target' column is present
        if 'target' not in df.columns:
            return jsonify({"error": "Missing 'target' column in input data"}), 400

        # Separate features and target
        X = df.drop(columns=['target'])
        y = df['target']

        # Predict using the model
        predictions = model.predict(X)

        # Check correctness
        correctness = (predictions == y).astype(int)

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y, predictions)

        # Build results
        results = pd.DataFrame({
            "Actual": y,
            "Predicted": predictions,
            "Correct(1)/Incorrect(0)": correctness
        })

        return jsonify({
            "overall_accuracy": round(overall_accuracy * 100, 2),
            "results": results.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if _name_ == '_main_':
    if train_model():
        app.run(debug=False)