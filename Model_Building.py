import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# read data set
diabetes_data = pd.read_csv("Data set/diabetes.csv", encoding= 'unicode_escape')

# define x, y
X = diabetes_data.drop(['class'], axis = 1)
y = diabetes_data['class']

# split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)


# ****************************************** Training Models *******************************************

# Naive Bayes
gnb = GaussianNB()
cv = cross_val_score(gnb, X_train, y_train, cv=5)
print(cv)
print(cv.mean())


# Logistic Regression
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr, X_train, y_train,cv=5)
print(cv)
print(cv.mean())


# Decision Tree
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, X_train, y_train, cv=5)
print(cv)
print(cv.mean())


# Random Forest
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf, X_train, y_train,cv=5)
print(cv)
print(cv.mean())
