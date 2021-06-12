from flask import Flask
from flask import jsonify
from flask import request

# Machine Learning Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)

@app.route('/', methods=['POST'])
def machineLearningAlog():

    dataset = pd.read_csv("heart.csv")

    data = request.get_json('data')


    age = data['age']
    sex = data['sex']
    cp = data['cp']
    trt_bps = data['trt_bps']
    chol = data['chol']
    fbs =data["fbs"]
    restecg = data["restecg"]
    thalachh = data["thalachh"]
    exng = data["exng"]
    old_peaks = data["old_peaks"]
    slp = data["slp"]
    cia = data["cia"]
    thall = data["thall"]
    itter = data["itter"]

    predictors = dataset.drop("output",axis=1)
    target = dataset["output"]
    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.1,random_state=0)

    max_accuracy = 0


    for x in range(itter):
        print(x)
        rf = RandomForestClassifier(random_state=x)
        rf.fit(X_train,Y_train)
        Y_pred_rf = rf.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
        if(current_accuracy>max_accuracy):
            max_accuracy = current_accuracy
            best_x = x
   
    rf = RandomForestClassifier(random_state=best_x)
    rf.fit(X_train,Y_train)

    Y_pred_rf = rf.predict([[age,sex,cp,trt_bps,chol,fbs,restecg,thalachh,exng,old_peaks,slp,cia,thall]])

    # Y_pred_rf = rf.predict([[56,1,0,126,249,1,0,144,1,1.2,1,1,2]])


    finalOutput = str(Y_pred_rf[0])


    return jsonify({"output":finalOutput})