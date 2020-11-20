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
| test | Insulin Test | Num |  |
| mass | Body Mass Index | Num |  |
| pedi | Pedigree | Num |  |
| age | Age | Num |   |
| class | Whether or not this person has diabetes | Num | 1 = Yes, 0 = No |


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
   preg  plas  pres  skin  test  mass   pedi  age  class
0     6   148    72    35     0  33.6  0.627   50      1
1     1    85    66    29     0  26.6  0.351   31      0
2     8   183    64     0     0  23.3  0.672   32      1
3     1    89    66    23    94  28.1  0.167   21      0
4     0   137    40    35   168  43.1  2.288   33      1
```
**Date column types and count**
```python
# understand the type of each column
print(diabetes_data.info())
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   preg    768 non-null    int64  
 1   plas    768 non-null    int64  
 2   pres    768 non-null    int64  
 3   skin    768 non-null    int64  
 4   test    768 non-null    int64  
 5   mass    768 non-null    float64
 6   pedi    768 non-null    float64
 7   age     768 non-null    int64  
 8   class   768 non-null    int64  
dtypes: float64(2), int64(7)
```
There are no null values.

**Summarize the central tendency, dispersion and shape**
```python
# get information on the numerical columns for the data set
with pd.option_context('display.max_columns', len(diabetes_data.columns)):
    print(diabetes_data.describe(include='all'))
```
```
             preg        plas        pres        skin        test        mass  \
count  768.000000  768.000000  768.000000  768.000000  768.000000  768.000000   
mean     3.845052  120.894531   69.105469   20.536458   79.799479   31.992578   
std      3.369578   31.972618   19.355807   15.952218  115.244002    7.884160   
min      0.000000    0.000000    0.000000    0.000000    0.000000    0.000000   
25%      1.000000   99.000000   62.000000    0.000000    0.000000   27.300000   
50%      3.000000  117.000000   72.000000   23.000000   30.500000   32.000000   
75%      6.000000  140.250000   80.000000   32.000000  127.250000   36.600000   
max     17.000000  199.000000  122.000000   99.000000  846.000000   67.100000   

             pedi         age       class  
count  768.000000  768.000000  768.000000  
mean     0.471876   33.240885    0.348958  
std      0.331329   11.760232    0.476951  
min      0.078000   21.000000    0.000000  
25%      0.243750   24.000000    0.000000  
50%      0.372500   29.000000    0.000000  
75%      0.626250   41.000000    1.000000  
max      2.420000   81.000000    1.000000  
```

## 4) Data Cleaning

### 4.1 Correcting outliers

#### 4.1.0 Helper methods
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

#### 4.1.1 preg (Pregnancies)
```python
sns.boxplot(x=diabetes_data['preg'])
plt.show()
```
<img src="/images/preg_boxplot.png" title="Pregnancies box plot" width="600" height="auto"/><br>

#### 4.1.2 plas (Plasma Glucose)
```python
sns.boxplot(x=diabetes_data['plas'])
plt.show()
```
<img src="/images/plas_boxplot.png" title="Plasma Glucose box plot" width="600" height="auto"/><br>

#### 4.1.3 skin (Skin Thickness)
```python
sns.boxplot(x=diabetes_data['skin'])
plt.show()
```
<img src="/images/skin_boxplot.png" title="Skin Thickness box plot" width="600" height="auto"/><br>

#### 4.1.4 test (Insulin Test)
```python
sns.boxplot(x=diabetes_data['test'])
plt.show()
```
<img src="/images/test_boxplot.png" title="Insulin Test box plot" width="600" height="auto"/><br>

#### 4.1.5 mass (Body Mass Index)
```python
sns.boxplot(x=diabetes_data['mass'])
plt.show()
```
<img src="/images/mass_boxplot.png" title="Body Mass Index box plot" width="600" height="auto"/><br>

#### 4.1.6 pedi (Pedigree)
```python
sns.boxplot(x=diabetes_data['pedi'])
plt.show()
```
<img src="/images/pedi_boxplot.png" title="Pedigree box plot" width="600" height="auto"/><br>

#### 4.1.7 age
```python
sns.boxplot(x=diabetes_data['age'])
plt.show()
```
<img src="/images/age_boxplot.png" title="Age box plot" width="600" height="auto"/><br>


## 5) Data Exploration
This section explores the distribution of each variable.

### 5.1 preg (Pregnancies)



### 5.19 Correlation Heatmap


## 6) Feature Engineering

### 6.1 Exploration of new features


### 6.2 Split into Training and Testing Data


## 7) Evaluate Model Performance

### 7.1 Data Preprocessing for Model


### 7.2 Model Building