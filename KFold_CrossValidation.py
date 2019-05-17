# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:24:33 2019

@author: saanvi
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

## ## KFold API usage example
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)

for train_index, test_index in kf.split(digits.data):
    print("KFold")
    
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

get_score(LogisticRegression(), X_train, X_test, y_train, y_test)
get_score(SVC(), X_train, X_test, y_train, y_test)
get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test)

## StratifiedKFold API usage example
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

lr_score = []
svm_score = []
rf_score = []

for train_index, test_index in kf.split(digits.data): 
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    lr_score.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    svm_score.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    rf_score.append(get_score(RandomForestClassifier(n_estimators=20), X_train, X_test, y_train, y_test))
    
print("Mean scopre of Logistic Regression", np.mean(lr_score))
print("Mean score of SVM Classifier", np.mean(svm_score))
print("Mean score of RandomForest Regression", np.mean(rf_score))


