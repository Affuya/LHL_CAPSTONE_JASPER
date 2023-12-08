from flask import Flask, redirect, request, render_template, url_for
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
    features = [
        int(request.form['Gender']),
        int(request.form['Married']),
        int(request.form['Education']),
        int(request.form['Self_Employed']),
        float(request.form['ApplicantIncome']),
        float(request.form['CoapplicantIncome']),
        float(request.form['LoanAmount']),
        float(request.form['Loan_Amount_Term']),
        float(request.form['Credit_History']),
        int(request.form['Property_Area'])
    ]

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([features], columns=[
        'Gender', 'Married', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ])

    # Set default values for additional features
    input_data['Total_Income'] = 0.0
    input_data['LoanAmount_to_TotalIncome_Ratio'] = 0.0
    input_data['LoanTerm_Multiplier'] = 0.0

    print("Features:", input_data.columns)

    # Make a prediction
    prediction = model.predict(input_data)

# Redirect to a new route for displaying the prediction result
    return redirect(url_for('result', prediction_text=prediction[0]))

@app.route('/result')
def result():
    # Retrieve the prediction result from the URL parameters
    prediction_text = request.args.get('prediction_text', 'No prediction available.')
    return render_template('result.html', prediction_text=f'Prediction: {prediction_text}')



if __name__ == '__main__':
    app.run(debug=True)
