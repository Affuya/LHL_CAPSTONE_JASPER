from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the loan approval model
loan_approval_model = load('loan_approval_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input features from the form
    gender = int(request.form['gender'])
    # Add similar lines for other form fields

    # Create a feature vector for prediction
    features = [gender]  # Add other features to this list

    # Make a prediction using the loan approval model
    prediction = loan_approval_model.predict([features])

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
