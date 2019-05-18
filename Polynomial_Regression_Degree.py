# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:10:52 2019

@author: saanvi
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2]
y = dataset.iloc[:,-1]

# Fitting Linear Regression to Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial Regression to dataset 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the Simple Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Simple Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')

# Visualizing the Polynomial Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Simple Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')

