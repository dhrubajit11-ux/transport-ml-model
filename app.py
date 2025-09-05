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
        # Get JSON data from request
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Predict using the model
        prediction = model.predict(input_data)

        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
