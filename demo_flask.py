##Flask app routing

from flask import Flask,request,jsonify
import joblib
import numpy as np
import tensorflow as tf
import keras
from PIL import Image


app=Flask(__name__)
@app.route("/",methods=["GET"])
def welcome():
    return "<h1>Welcome to the ml model api</h1>"

model=joblib.load("model.pkl")
@app.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    features=np.array(list(data.values()),dtype=float).reshape(1,-1)
    prediction=model.predict(features)
    return jsonify({"prediction:":int(prediction[0])})

model1=joblib.load("model1.pkl")
@app.route("/classify",methods=["POST"])
def classify():
    data=request.get_json()
    features=np.array(list(data.values()),dtype=float).reshape(1,-1)
    classify=model1.predict(features)
    return jsonify({"Classification:":int(classify[0])})

model2=keras.models.load_model("tensor.keras")
@app.route("/pred",methods=["POST"])
def pred():
    file=request.files['image']
    img=Image.open(file).convert('L')
    img=img.resize((28,28))
    img_array=np.array(img)/255.0
    img_array=img_array.reshape(1,28,28)
    pred=model2.predict(img_array)
    digit=int(np.argmax(pred))
    return jsonify({"prediction:":digit,"confidence": float(np.max(pred))})

if __name__=="__main__":
    app.run(debug=True)
