from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load scaler and model
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
model = tf.keras.models.load_model("model (2).h5")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        data = request.json['features']  # Assuming 'features' key in JSON payload
        data = np.array(data).reshape(1, -1)  # Ensure the correct shape
        
        # Preprocess and predict
        data_scaled = scaler.transform(data)
        data_pca = pca.transform(data_scaled)
        predictions = model.predict(data_pca)
        
        return jsonify({'probabilities': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
