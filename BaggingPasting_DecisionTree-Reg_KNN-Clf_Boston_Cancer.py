# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:52:06 2019
This program explans the bagging regressor
@author: saanvi
"""


"""
Method 1::==> Bagging 
"""

# Load data from sklearn
from sklearn.datasets import load_boston
dataset = load_boston()
X = dataset.data
y = dataset.target

# Generating training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Use bagging regressor with DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
bag_tree = BaggingRegressor(DecisionTreeRegressor(), max_features=0.7, n_estimators=100, random_state=3)

# Fit and predict the result
bag_tree.fit(X_train, y_train)
y_pred = bag_tree.predict(X_test)

# Evaluate the model
print(bag_tree.score(X_test, y_test))


"""
Method 2::==> pasting 
"""

# Load data from sklearn
from sklearn.datasets import load_boston
dataset = load_boston()
X = dataset.data
y = dataset.target

# Generating training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Use bagging regressor with DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
bag_tree = BaggingRegressor(DecisionTreeRegressor(), max_samples=0.8, n_estimators=100, bootstrap=False, random_state=3)

# Fit and predict the result
bag_tree.fit(X_train, y_train)
y_pred = bag_tree.predict(X_test)

# Evaluate the model
bag_tree.score(X_test, y_test)



"""
This program explans the bagging regressor

Here, we are using cancer dataset
"""

"""
Method 1::==> Bagging 
"""

# Load data from sklearn
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Generating training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Use bagging classifier with KNN
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bag_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_features=0.8, n_estimators=100, random_state=3)

# Fit and predict the result
bag_knn.fit(X_train, y_train)
bag_knn.score(X_test, y_test)


"""
Method 2::==> pasting 
"""

# Load data from sklearn
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Generating training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Use bagging classifier with KNN
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bag_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_samples=0.8, n_estimators=100, bootstrap=False, random_state=3)

# Fit and predict the result
bag_knn.fit(X_train, y_train)
bag_knn.score(X_test, y_test)

