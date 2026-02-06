from flask import Flask, render_template, request
import pandas as pd
import pickle as p
import numpy as np

app = Flask(__name__)

# Load the heart disease prediction model
model = p.load(open('heart_model.pkl', 'rb'))

# Load the scaler for data normalization
scaler = p.load(open('scaler.pkl', 'rb'))

# Load the feature names for reference
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']

# Define the sample page route
@app.route('/')
def sample_page():
    return render_template('home.html')

# Define the input field page route
@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        # Get the form inputs from the request
        form_inputs = [float(request.form[name]) for name in feature_names]

        # Normalize the input data
        normalized_inputs = scaler.transform([np.array(form_inputs)])

        # Make the prediction
        prediction = model.predict(normalized_inputs)[0]

        # Map the prediction to a human-readable label
        prediction_label = 'Presence of Heart Disease' if prediction == 1 else 'Absence Of Heart Disease'

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction_label)
    else:
        return render_template('index.html', feature_names=feature_names)

if __name__ == '__main__':
    # Load the heart disease dataset
    data = pd.read_csv('heart.csv')

    # Separate the features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Normalize the feature data
    X_normalized = scaler.fit_transform(X)

    # Train the model
    model.fit(X_normalized, y)

    app.run(debug=True)