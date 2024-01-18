# app.py

from flask import Flask, request, jsonify
#import xgboost as xgb
import pickle

app = Flask(__name__)

# Load the XGBoost model
model_path = "C:\\Users\\affuy\\Documents\\LHL\\LHL_CAPSTONE_JASPER\\best_xgb_model.p"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        # Assuming the input features are in 'features' key
        features = data['features']

        # Convert features to a format compatible with your model
        # Make predictions
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
