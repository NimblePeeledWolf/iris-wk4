from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Fetch Iris dataset
iris = fetch_ucirepo(id=53) 
  
x = iris.data.features 
y = iris.data.targets 
y = y.values.ravel()

# Encode target labels (class names) into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y = y_encoded
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Load the pre-trained SVM model, scaler, and label encoder
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])  # Added missing feature
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare the features for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale the features using the scaler
        scaled_features = scaler.transform(features)

        # Make the prediction using the trained SVM model
        prediction = model.predict(scaled_features)

        # Map the numeric prediction to the class name using the label encoder
        predicted_class = label_encoder.inverse_transform(prediction)

        # Return the prediction to the user by passing it to the template
        return render_template('index.html', prediction=predicted_class[0])

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
