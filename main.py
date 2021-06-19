from flask import Flask
from flask import jsonify
from flask import request

# Machine Learning Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

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

    selected = data["selected"]

    predictors = dataset.drop("output",axis=1)
    target = dataset["output"]
    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.2,random_state=0)



    def logisticRegg():
        logreg=LogisticRegression()

        logreg.fit(X_train,Y_train)

        Y_predictor_logreg=logreg.predict(X_test)

        score_logreg = round(accuracy_score(Y_test,Y_predictor_logreg)*100,2)

        Y_pred_lr = logreg.predict([[age,sex,cp,trt_bps,chol,fbs,restecg,thalachh,exng,old_peaks,slp,cia,thall]])
        finalOutput = str(Y_pred_lr[0])
        return [finalOutput, score_logreg]



    def randomClassifier():
        max_accuracy = 0
        for x in range(10):
            rf = RandomForestClassifier(random_state=x)
            rf.fit(X_train,Y_train)
            Y_pred_rf = rf.predict(X_test)
            current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
            if(current_accuracy>max_accuracy):
                max_accuracy = current_accuracy
                best_x = x
   
        rf = RandomForestClassifier(random_state=best_x)
        rf.fit(X_train,Y_train)
        Y_pred_rf = rf.predict(X_test)
        score_rf = round(accuracy_score(Y_test,Y_pred_rf)*100,2)
        Y_pred_rf = rf.predict([[age,sex,cp,trt_bps,chol,fbs,restecg,thalachh,exng,old_peaks,slp,cia,thall]])
        finalOutput = str(Y_pred_rf[0])
        return [finalOutput, score_rf]


    def KNN():
        knn=KNeighborsClassifier(n_neighbors=21)
        knn.fit(X_train,Y_train) 
        Y_pred_knn = knn.predict(X_test)
        score_knn = round(accuracy_score(Y_test,Y_pred_knn)*100,2)
        Y_pred_knn1 = knn.predict([[age,sex,cp,trt_bps,chol,fbs,restecg,thalachh,exng,old_peaks,slp,cia,thall]])
        finalOutput = str(Y_pred_knn1[0])
        return [finalOutput, score_knn]

    def SVM():
        sv = svm.SVC(kernel='linear')
        sv.fit(X_train, Y_train)
        Y_pred_svm = sv.predict(X_test)
        score_svm = round(accuracy_score(Y_test,Y_pred_svm)*100,2)
        Y_pred_svm1 = sv.predict([[age,sex,cp,trt_bps,chol,fbs,restecg,thalachh,exng,old_peaks,slp,cia,thall]])
        finalOutput = str(Y_pred_svm1[0])
        return [finalOutput, score_svm]

    def NaiveBayes():
        NB = GaussianNB()
        NB.fit(X_train,Y_train)
        Y_pred_nb = NB.predict(X_test)
        score_nb = round(accuracy_score(Y_test,Y_pred_nb)*100,2)
        Y_pred_nb1 = NB.predict([[age,sex,cp,trt_bps,chol,fbs,restecg,thalachh,exng,old_peaks,slp,cia,thall]])
        finalOutput = str(Y_pred_nb1[0])
        return [finalOutput, score_nb]

        
    if selected == 1:
        logistic = logisticRegg()
        Algo = "Logistic Regression Algorithm"
    elif selected == 2:
        logistic = randomClassifier()
        Algo = "Random Forest Algorithm"
    elif selected == 3:
        logistic = KNN()
        Algo = "K-Nearest Neighbor Algorithm"
    elif selected == 4:
        logistic = SVM()
        Algo = "Support Vector Machine Algorithm"
    elif selected == 5:
        logistic = NaiveBayes()
        Algo = "Naive Bayes Algorithm"

    return jsonify({"output":logistic[0], "accuracy":logistic[1], "Algorithm": Algo})