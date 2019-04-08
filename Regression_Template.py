# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 08:32:47 2019

@author: ab59349

Standard Regression Template
"""

# Importing the libraries and dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Taking care of  missing data
"""from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X.iloc[:,1:3])
X.iloc[:,1:3] = imputer.transform(X.iloc[:,1:3]).astype(int)"""

# Encoding categorical data
# Encoding independent variable
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Avoiding dummy variable trap
X = X[:, 1:]"""

# Import SimpleLinearRegression from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model to training data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction using the model on test dataset
y_pred = regressor.predict(X_test)


# Check model accuracy using r2_score
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)



