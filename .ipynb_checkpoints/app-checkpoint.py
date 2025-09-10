from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load("transport_model.pkl")

@app.route('/')
def home():
    return "Transport Mode Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Get prediction and probability
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Find the index of the predicted class
        predicted_class_index = list(model.classes_).index(prediction[0])

        # Get confidence score (as percentage)
        confidence = prediction_proba[0][predicted_class_index] * 100

        return jsonify({
            "results": [
                {
                    "Prediction": prediction[0],
                    "Accuracy": f"{confidence:.2f}%"
                }
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
