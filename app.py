from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the saved model
model = joblib.load("transport_model.pkl")

@app.route('/')
def home():
    return "🚀 Transport Mode Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
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


if __name__ == "__main__":
    app.run(debug=True)
