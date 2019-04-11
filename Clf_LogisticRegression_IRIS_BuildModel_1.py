# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:42:08 2019

@author: saanvi
"""
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Import dataset
col_name = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=col_name)
dataset['species'] = dataset['species'].str.split('-').str.get(1)
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Handeling categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Generating training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Save model
saved_model = pickle.dumps(classifier)
from_pickle = pickle.loads(saved_model)

# Prediction
y_pred = from_pickle.predict(X_test)

# Accuracy check 
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

