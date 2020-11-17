# Store Prediction

## Technology and Resources Used

**Python Version**: 3.7.7

## Table of Contents
1) [Define the Problem](#1-define-the-problem)<br>
2) [Gather the Data](#2-gather-the-data)
3) [Prepare Data for Consumption](#3-prepare-data-for-consumption)<br>
4) [Data Cleaning](#4-data-cleaning)<br>
5) [Data Exploration](#5-data-exploration)<br>
6) [Feature Engineering](#6-feature-engineering)<br>
7) [Model Building](#7-model-building)<br>

## 1) Define the Problem
The mandate is to predict if a person has diabetes or not.

## 2) Gather the Data
The data sets were provided. They are uploaded in the data sets folder.

## 3) Prepare Data for Consumption

### 3.1 Import Libraries
The following code is written in Python 3.7.7. Below is the list of libraries used.
```python
import numpy as np 
import pandas as pd
import matplotlib
import sklearn
import itertools
import copy
import csv
import openpyxl
```

### 3.2 Load Data Modeling Libraries
These are the most common machine learning and data visualization libraries.
```python
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Common Model Algorithms
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
```

### 3.3 Data dictionary
The data dictionary for the data set is as follows:<br>
| Variable | Definition | Type | Key/Sample values|
| :-------: | :---------:| :-------:| :-------:|
| preg | Pregnancies | Num |  |
| plas | Plasma Glucose | Num |  |
| pres | Blood Pressure | Num |  |
| skin | Skin Thickness | Num |  |
| test | Insulin test | Num |  |
| mass | Body Mass Index | Num |  |
| pedi | Pedigree | Num |  |
| age | Person's age | Num |   |
| class | Product Hierarchy Level 20 Code | Char | 1 = Yes, 0 = No |


### 3.5 Greet the data
**Import data**
```python
# read data set
diabetes_data = pd.read_csv("Data set/diabetes.csv", encoding= 'unicode_escape')
```
**Preview data**
```python
# get a peek at the top 5 rows of the data set
print(diabetes_data.head())
```
```

```
**Date column types and count**
```python
# understand the type of each column
print(diabetes_data.info())
```
```

```


**Summarize the central tendency, dispersion and shape**
```python
# get information on the numerical columns for the data set
with pd.option_context('display.max_columns', len(diabetes_data.columns)):
    print(diabetes_data.describe(include='all'))
```
```

```

## 4) Data Cleaning
The data is cleaned in 2 steps:
1. Correcting outliers
2. Completing null or missing data
3. Create new features
4. Convert object data types
5. Output cleaned data into new CSV

### 4.1 Correcting outliers

#### Helper methods
These 2 methods were created to find the outliers and remove them from a column.<br>
**IQR method**
```python
def remove_outliers_iqr(df):
	dataf = pd.DataFrame(df)
	quartile_1, quartile_3 = np.percentile(dataf, [25,75])

	iqr = quartile_3 - quartile_1
	lower_bound = quartile_1 - (iqr * 1.5)
	upper_bound = quartile_3 + (iqr * 1.5)

	print("lower bound:", lower_bound)
	print("upper bound:", upper_bound)
	print("IQR outliers:", np.where((dataf > upper_bound) | (dataf < lower_bound)))
	print("# of outliers:", len(np.where((dataf > upper_bound) | (dataf < lower_bound))[0]))

	return dataf[~((dataf < lower_bound) | (dataf > upper_bound)).any(axis=1)]
```
**Z-Score method**
```python
def remove_outliers_z_score(df):
	threshold = 3
	dataf = pd.DataFrame(df)
	z_scores = np.abs(stats.zscore(dataf))

	print("z-score outliers:", np.where(z_scores > threshold))
	print("# of outliers:", len(np.where(z_scores > threshold)[0]))
	
	return dataf[(z_scores  < 3).all(axis=1)]
```

#### 4.1.1 QUANTITY


### 4.2 Completing null or missing data



#### 4.2.5 Drop all null rows


### 4.3 Create new features


### 4.4 Convert object data types


### 4.5 Output cleaned data into new CSV

```python
diabetes_data.to_csv('Data set/diabetes_data_cleaned.csv',index = False)
```

## 5) Data Exploration
This section explores the distribution of each variable.
```python
# read data set
diabetes_data = pd.read_csv("Data set/diabetes_data_cleaned.csv", encoding= 'unicode_escape')
```


### 5.19 Correlation Heatmap


## 6) Feature Engineering

### 6.1 Exploration of new features


### 6.2 Split into Training and Testing Data


## 7) Evaluate Model Performance

### 7.1 Data Preprocessing for Model


### 7.2 Model Building