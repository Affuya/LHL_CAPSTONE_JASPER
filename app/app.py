from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model
with open("../best_rf_model.p", "rb") as file:
    model = pickle.load(file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [request.form['Gender'],
                request.form['Married'],
                request.form['Education'],
                request.form['Self_Employed'],
                int(request.form['ApplicantIncome']),
                int(request.form['CoapplicantIncome']),
                int(request.form['LoanAmount']),
                int(request.form['Loan_Amount_Term']),
                float(request.form['Credit_History']),
                request.form['Property_Area']]

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([features], columns=[
        'Gender', 'Married', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ])

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the result
    return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
