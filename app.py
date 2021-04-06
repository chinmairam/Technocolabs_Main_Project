from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model1 = pickle.load(open('lending_loan.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    predictions = model1.predict(final)
    output = predictions[0]
    #return render_template('predict.html', prediction_text="Prediction is {}".format(output))

    if output == 0:
        return render_template('predict.html', prediction_text=f'Sorry,Your loan was not approved')
    else:
        return render_template('predict.html', prediction_text=f'Your loan was approved')

if __name__ == '__main__':
    app.run(debug=True)
