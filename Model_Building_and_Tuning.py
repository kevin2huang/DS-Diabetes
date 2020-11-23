import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# read data set
diabetes_data = pd.read_csv("Data set/diabetes_data_cleaned.csv", encoding= 'unicode_escape')

# define x, y
X = diabetes_data.drop(['class'], axis = 1)
y = diabetes_data['class']

# split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)


# ****************************************** Training Models *******************************************

# Logistic Regression
# lr = LogisticRegression(max_iter = 2000)
# cv = cross_val_score(lr, X_train, y_train,cv=5)
# print(cv)
# print(cv.mean())


# Decision Tree
# dt = tree.DecisionTreeClassifier(random_state = 1)
# cv = cross_val_score(dt, X_train, y_train, cv=5)
# print(cv)
# print(cv.mean())


# Random Forest
# rf = RandomForestClassifier(random_state = 1)
# cv = cross_val_score(rf, X_train, y_train,cv=5)
# print(cv)
# print(cv.mean())


# ****************************************** Hyperparameter Tuning *******************************************


# Logistic Regression



# Decision Tree
# gini_acc_scores = []
# entropy_acc_scores = []

# criterions = ["gini", "entropy"]

# for criterion in criterions:
# 	for depth in range(25):
# 	    dt = tree.DecisionTreeClassifier(criterion=criterion, max_depth = depth+1, random_state=depth)
# 	    model = dt.fit(X_train,y_train)
	    
# 	    y_predict = dt.predict(X_test)

# 	    if criterion == "gini":
# 	    	gini_acc_scores.append(accuracy_score(y_test, y_predict))
# 	    else:
# 	    	entropy_acc_scores.append(accuracy_score(y_test, y_predict))


# # plot the accuracy scores by depths and criterion
# figuresize = plt.figure(figsize=(12,8))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# EntropyAcc = plt.plot(np.arange(25)+1, entropy_acc_scores, '--bo')   
# GiniAcc = plt.plot(np.arange(25)+1, gini_acc_scores, '--ro')
# legend = plt.legend(['Entropy', 'Gini'], loc ='lower right',  fontsize=15)
# title = plt.title('(Decision Tree) Accuracy Score for Multiple Depths', fontsize=25)
# xlab = plt.xlabel('Depth of Tree', fontsize=20)
# ylab = plt.ylabel('Accuracy Score', fontsize=20)

# plt.show()

# print("Gini max accuracy:", max(gini_acc_scores))
# print("Entropy max accuracy:", max(entropy_acc_scores))


# # use best depth for prediction
# dt = tree.DecisionTreeClassifier(max_depth = 1, random_state = 1)
# dt = dt.fit(X_train, y_train)
# y_predict = dt.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_predict))


# RandomForest
acc_scores = []              
depth = np.arange(1, 30)

for i in depth:

    rf = RandomForestClassifier(n_estimators=25, max_depth=i, random_state=1)
    rf.fit(X_train,y_train)

    y_predict = rf.predict(X_test)

    acc_scores.append(accuracy_score(y_test, y_predict)) 


figsize = plt.figure(figsize = (12,8))
plot = plt.plot(depth, acc_scores, 'r')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
xlab = plt.xlabel('Depth of the trees', fontsize = 20)
ylab = plt.ylabel('Accuracy', fontsize = 20)
title = plt.title('(Random Forest) Accuracy vs Depth of Trees', fontsize = 25)
plt.show()


rf = RandomForestClassifier(n_estimators=25, max_depth=acc_scores.index(max(acc_scores))+1, random_state=1)
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predict))