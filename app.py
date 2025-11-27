from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and LabelEncoder
model = joblib.load('random_forest_model.pkl')          # Trained model
scaler = joblib.load('scaler.pkl')                      # Scaler used during training
label_encoder = joblib.load('size_encoder.pkl')         # LabelEncoder used for target

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json()

    # Extract values from input
    fabric_type = data.get('type')
    chest = data.get('chest')
    front = data.get('front')
    shoulder = data.get('shoulder')

    # Check if all inputs are present
    if None in [fabric_type, chest, front, shoulder]:
        return jsonify({'error': 'Missing input values'}), 400

    # Prepare the input array for scaling and prediction
    input_data = np.array([[fabric_type, chest, front, shoulder]])
    input_scaled = scaler.transform(input_data)

    # Predict the numeric label
    numeric_prediction = model.predict(input_scaled)[0]

    # Convert numeric label to actual size
    predicted_size = label_encoder.inverse_transform([numeric_prediction])[0]

    return jsonify({'size': predicted_size})

if __name__ == '__main__':
    app.run(debug=True)
