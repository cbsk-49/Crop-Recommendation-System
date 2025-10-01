from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

# Initialize Flask App
app = Flask(__name__)

# Load Pre-trained Objects
model = pickle.load(open('stacked_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# --- Routes ---

# Route to render the HTML form
@app.route('/')
def home():
    """Serves the frontend web page."""
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Receives input data, makes a prediction, and returns the crop."""
    try:
        data = request.get_json(force=True)
        
        # The order of features must match the training data
        features = [
            data['N'], data['P'], data['K'], data['temperature'],
            data['humidity'], data['ph'], data['rainfall']
        ]
        
        # Create DataFrame with proper feature names to avoid warnings
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features_df = pd.DataFrame([features], columns=feature_names)
        
        scaled_features = scaler.transform(features_df)
        
        prediction_encoded = model.predict(scaled_features)
        prediction_label = le.inverse_transform(prediction_encoded)
        
        return jsonify({'recommended_crop': prediction_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the App
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)