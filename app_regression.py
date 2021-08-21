# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:58:09 2021

@author: anann
"""

""" STEPS """
# 1. build the model 
# 2. Save all the dependencies
# 3. Create app.py file: load all dependencies for prediction
# 4. showcase your output on UI

from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
import joblib

# load the regression model
model = load_model("profit.h5")
# load column tranform
ct = joblib.load("column")
# load standardize
sc = joblib.load("scaler")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Regression.html")

@app.route('/Login', methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        ms = request.form["ms"]
        ad = request.form["as"]
        rd = request.form["rd"]
        s = request.form["s"]
        
        data = [[ms,ad,rd,s]]
        
        pred = model.predict(sc.transform(ct.transform(data)))
        print("Predicted Profit: ", pred)
    
    # To print the values in the string format
    # [0][0] is for removing the list format
    return render_template("Regression.html", value = str(pred[0][0]))

if __name__ == "__main__":
    app.run(debug = True)