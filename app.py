import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.externals import joblib
from flask import Flask, request, render_template


app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(map(float, feature_list.values()))
    final_features = np.array(feature_list).reshape(1, -1) 
    
    prediction = model.predict(final_features)
    output = int(prediction[0])
    if output == 1:
        text = "Diabetes"
    else:
        text = "Normal"

    return render_template('index.html', prediction_text='La predicción es: {}'.format(text))

if __name__ == "__main__":
    app.run(debug=True)
