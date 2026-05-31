import pickle
from flask import Flask,request,render_template,app,url_for,jsonify
import numpy as np
import pandas as pd
from django.shortcuts  import render

app= Flask(__name__)
##load the model
regmodel=pickle.load(open("regmodel.pkl","rb"))
scaling=pickle.load(open("scaling.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data=request.get_json()['data']
    print(data)
    # Assuming the input data has the same feature names as used during training
    input_data = np.array(list(data.values())).reshape(1, -1)
    new_data=scaling.transform(input_data)
    prediction = regmodel.predict(new_data)
    return jsonify({'prediction': prediction[0]})


if __name__=="__main__":
  app.run(debug=True)  