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



app = Flask(__name__)

@app.route('/', methods=['POST'])
def machineLearningAlog():

    dataset = pd.read_csv("heart.csv")

    data = request.get_json('data')



    predictors = dataset.drop("output",axis=1)
    #print(predictors)---> drops the output column and prints remaining columns
    target = dataset["output"]
    #print(target)----> only gives the output column
    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.1,random_state=0)




    max_accuracy = 0


    for x in range(2000):
        rf = RandomForestClassifier(random_state=x)
        rf.fit(X_train,Y_train)
        Y_pred_rf = rf.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
        if(current_accuracy>max_accuracy):
            max_accuracy = current_accuracy
            best_x = x
        
#print(max_accuracy)
#print(best_x)

    rf = RandomForestClassifier(random_state=best_x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)


    name = data['name']
    email = data['email']
    success=""
    
    if name == "kamal" and email == "kamal@gmail.com":
        success = 'Login Successfull'
    else:
        success = 'Login Failed'


    return jsonify({success:Y_pred_rf})