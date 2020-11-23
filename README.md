# Diabetes Prediction

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
8) [Hyperparameter Tuning](#8-hyperparameter-tuning)<br>

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
```

### 3.2 Load Data Modeling Libraries
These are the most common machine learning and data visualization libraries.
```python
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Common Model Algorithms
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Common Model Helpers
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
```

### 3.3 Data dictionary
The data dictionary for the data set is as follows:<br>
| Variable | Definition | Type | Key|
| :-------: | :---------:| :-------:| :-------:|
| preg | Pregnancies | Numerical |  |
| plas | Plasma Glucose Levels (mg/dL) | Numerical |  |
| pres | Blood Pressure | Numerical |  |
| skin | Skin Thickness | Numerical |  |
| test | Insulin Level | Numerical |  |
| mass | Body Mass Index | Numerical |  |
| pedi | Pedigree | Numerical |  |
| age | Age | Numerical |   |
| class | Whether or not this person has diabetes | Categorical | 1 = Yes, 0 = No |


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

#### 4.1.0 IQR method
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

#### 4.1.1 preg (Pregnancies)
```python
sns.boxplot(x=diabetes_data['preg'])
plt.show()
```
<img src="/images/preg_boxplot.png" title="Pregnancies box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['preg'] = remove_outliers_iqr(diabetes_data['preg'])
```
```
lower bound: -6.5
upper bound: 13.5
IQR outliers: (array([ 88, 159, 298, 455], dtype=int64), array([0, 0, 0, 0], dtype=int64))
# of outliers: 4
```

#### 4.1.2 plas (Plasma Glucose)
```python
sns.boxplot(x=diabetes_data['plas'])
plt.show()
```
<img src="/images/plas_boxplot.png" title="Plasma Glucose box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['plas'] = remove_outliers_iqr(diabetes_data['plas'])
```
```
lower bound: 37.125
upper bound: 202.125
IQR outliers: (array([ 75, 182, 342, 349, 502], dtype=int64), array([0, 0, 0, 0, 0], dtype=int64))
# of outliers: 5
```

#### 4.1.3 skin (Skin Thickness)
```python
sns.boxplot(x=diabetes_data['skin'])
plt.show()
```
<img src="/images/skin_boxplot.png" title="Skin Thickness box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['skin'] = remove_outliers_iqr(diabetes_data['skin'])
```
```
lower bound: -48.0
upper bound: 80.0
IQR outliers: (array([579], dtype=int64), array([0], dtype=int64))
# of outliers: 1
```

#### 4.1.4 test (Insulin Test)
```python
sns.boxplot(x=diabetes_data['test'])
plt.show()
```
<img src="/images/test_boxplot.png" title="Insulin Test box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['test'] = remove_outliers_iqr(diabetes_data['test'])
```
```
lower bound: -190.875
upper bound: 318.125
IQR outliers: (array([  8,  13,  54, 111, 139, 153, 186, 220, 228, 231, 247, 248, 258,
       286, 296, 360, 370, 375, 392, 409, 415, 480, 486, 519, 574, 584,
       612, 645, 655, 695, 707, 710, 715, 753], dtype=int64), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64))
# of outliers: 34
```

#### 4.1.5 mass (Body Mass Index)
```python
sns.boxplot(x=diabetes_data['mass'])
plt.show()
```
<img src="/images/mass_boxplot.png" title="Body Mass Index box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['mass'] = remove_outliers_iqr(diabetes_data['mass'])
```
```
lower bound: 13.35
upper bound: 50.550000000000004
IQR outliers: (array([  9,  49,  60,  81, 120, 125, 145, 177, 193, 247, 303, 371, 426,
       445, 494, 522, 673, 684, 706], dtype=int64), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int64))
# of outliers: 19
```

#### 4.1.6 pedi (Pedigree)
```python
sns.boxplot(x=diabetes_data['pedi'])
plt.show()
```
<img src="/images/pedi_boxplot.png" title="Pedigree box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['pedi'] = remove_outliers_iqr(diabetes_data['pedi'])
```
```
lower bound: -0.32999999999999996
upper bound: 1.2
IQR outliers: (array([  4,  12,  39,  45,  58, 100, 147, 187, 218, 228, 243, 245, 259,
       292, 308, 330, 370, 371, 383, 395, 445, 534, 593, 606, 618, 621,
       622, 659, 661], dtype=int64), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0], dtype=int64))
# of outliers: 29
```

#### 4.1.7 age
```python
sns.boxplot(x=diabetes_data['age'])
plt.show()
```
<img src="/images/age_boxplot.png" title="Age box plot" width="400" height="auto"/><br>
```python
# calculate IQR score and remove outliers
diabetes_data['age'] = remove_outliers_iqr(diabetes_data['age'])
```
```
lower bound: -1.5
upper bound: 66.5
IQR outliers: (array([123, 363, 453, 459, 489, 537, 666, 674, 684], dtype=int64), array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64))
# of outliers: 9
```

#### 4.1.8 drop null values
```python
diabetes_data = diabetes_data.dropna()

print(diabetes_data.info())
```
```
Int64Index: 673 entries, 0 to 767
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   preg    673 non-null    float64
 1   plas    673 non-null    float64
 2   pres    673 non-null    int64  
 3   skin    673 non-null    float64
 4   test    673 non-null    float64
 5   mass    673 non-null    float64
 6   pedi    673 non-null    float64
 7   age     673 non-null    float64
 8   class   673 non-null    int64  
dtypes: float64(7), int64(2)
```

#### 4.1.9 Convert object data type
```python
diabetes_data['preg'] = diabetes_data['preg'].astype(int)
diabetes_data['plas'] = diabetes_data['plas'].astype(int)
diabetes_data['pres'] = diabetes_data['pres'].astype(int)
diabetes_data['skin'] = diabetes_data['skin'].astype(int)
diabetes_data['test'] = diabetes_data['test'].astype(int)
diabetes_data['age'] = diabetes_data['age'].astype(int)
diabetes_data['class'] = diabetes_data['class'].astype(int)

diabetes_data['mass'] = diabetes_data['mass'].astype(float)
diabetes_data['pedi'] = diabetes_data['pedi'].astype(float)
```

#### 4.1.10 Output to CSV
Output cleaned data to CSV.
```python
diabetes_data.to_csv('Data set/diabetes_data_cleaned.csv',index = False)
```

## 5) Data Exploration
This section explores the distribution of each variable using cleaned data set.
```python
diabetes_data = pd.read_csv("Data set/diabetes_data_cleaned.csv", encoding= 'unicode_escape')
```

### 5.0 Helper method
```python
def plotHist(xlabel, title, column):
    fig, ax = plt.subplots(1, 1, 
                           figsize =(10, 7),  
                           tight_layout = True)

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)

    plt.xlabel(xlabel, fontsize=16)  
    plt.ylabel("# of entries", fontsize=16)
    plt.title(title, fontsize=20)

    plt.hist(column)
    plt.show()
```
```python
def plotBar(xlabel, title, column):
    ax = sns.barplot(column.value_counts().index, column.value_counts())

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)

    plt.xlabel(xlabel, fontsize=16)  
    plt.ylabel("# of entries", fontsize=16)
    plt.title(title, fontsize=20)

    plt.show()
```
```python
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9}, 
        ax=ax,
        annot=True, 
        linewidths=0.1, 
        vmax=1.0, 
        linecolor='white',
        annot_kws={'fontsize':14}
    )

    _.set_yticklabels(_.get_ymajorticklabels(), fontsize = 16)
    _.set_xticklabels(_.get_xmajorticklabels(), fontsize = 16)

    plt.title('Pearson Correlation of Features', y=1.05, size=20)

    plt.show()
```
### 5.1 preg (Pregnancies)

```python
print('preg (Pregnancies):\n', diabetes_data.preg.value_counts(sort=False))
plotHist("Pregnancies", "Histogram of number of entries per number of pregnancies", diabetes_data.preg) 
```
<img src="/images/preg_hist.png" title="Histogram of number of entries per pregnancy count" width="500" height="auto"/><br>
```
preg (Pregnancies):
0      97
1     118
2      89
3      68
4      64
5      49
6      44
7      40
8      31
9      24
10     22
11      9
12      8
13     10
Name: preg, dtype: int64
```

### 5.2 plas (Plasma Glucose)
```python
print('plas (Plasma Glucose):\n', diabetes_data.plas.value_counts(sort=False))
plotHist("Plasma Glucose Levels (mg/dL)", "Histogram of number of entries per plasma glucose levels", diabetes_data.plas) 
```
<img src="/images/plas_hist.png" title="Histogram of number of entries per plasma glucose score" width="500" height="auto"/><br>
```
plas (Plasma Glucose):
44     1
56     1
57     1
61     1
62     1
      ..
194    2
195    2
196    3
197    1
198    1
Name: plas, Length: 132, dtype: int64
```

### 5.3 skin
```python
print('skin (Skin Thickness):\n', diabetes_data.skin.value_counts(sort=False))
plotHist("Skin Thickness", "Histogram of number of entries per skin thickness length", diabetes_data.skin) 
```
<img src="/images/skin_hist.png" title="Histogram of number of entries per skin thickness score" width="500" height="auto"/><br>
```
skin (Skin Thickness):
0     206
7       1
8       2
10      5
11      6
12      7
13     10
14      5
15     13
16      5
17     14
18     18
19     16
20     10
21      9
22     13
23     19
24      8
25     15
26     15
27     22
28     19
29     15
30     23
31     18
32     29
33     14
34      8
35     10
36     13
37     14
38      6
39     16
40     16
41     11
42      6
43      4
44      3
45      5
46      7
47      3
48      3
49      2
50      3
51      1
52      2
54      2
60      1
Name: skin, dtype: int64
```

### 5.4 test
```python
print('test (Insulin Level):\n', diabetes_data.test.value_counts(sort=False))
plotHist("Insulin Level", "Histogram of number of entries per insulin level", diabetes_data.test)
```
<img src="/images/test_hist.png" title="Histogram of number of entries per insulin level" width="500" height="auto"/><br>
```
test (Insulin Level):
0      338
15       1
16       1
18       2
22       1
      ... 
293      1
300      1
304      1
310      1
318      1
Name: test, Length: 150, dtype: int64
```

### 5.5 mass
```python
print('mass (Body Mass Index):\n', diabetes_data.mass.value_counts(sort=False))
plotHist("Body Mass Index", "Histogram of number of entries per body mass index score", diabetes_data.mass) 
```
<img src="/images/mass_hist.png" title="Histogram of number of entries per body mass index score" width="500" height="auto"/><br>
```
mass (Body Mass Index):
31.0     2
38.0     2
30.0     6
29.0     4
36.0     2
        ..
26.9     1
36.6     4
23.4     1
46.3     1
31.2    10
Name: mass, Length: 232, dtype: int64
```

### 5.6 pedi
```python
print('pedi (Pedigree):\n', diabetes_data.pedi.value_counts(sort=False))
plotHist("Pedigree", "Histogram of number of entries per pedigree count", diabetes_data.mass) 
```
<img src="/images/pedi_hist.png" title="Histogram of number of entries per pedigree count" width="500" height="auto"/><br>
```
pedi (Pedigree):
0.375    1
0.875    2
0.560    1
0.381    1
0.514    2
        ..
0.347    1
0.236    3
0.231    2
0.893    1
0.084    1
Name: pedi, Length: 461, dtype: int64
```

### 5.7 age
```python
print('age:\n', diabetes_data.age.value_counts(sort=False))
plotHist("Age", "Histogram of number of entries per age", diabetes_data.age) 
```
<img src="/images/age_hist.png" title="Histogram of number of entries per age" width="500" height="auto"/><br>
```
age:
21    56
22    63
23    35
24    42
25    38
26    29
27    31
28    31
29    27
30    19
31    22
32    15
33    13
34    10
35     9
36    16
37    18
38    15
39    12
40    11
41    21
42    17
43    11
44     7
45    14
46     9
47     5
48     5
49     4
50     7
51     7
52     7
53     4
54     5
55     4
56     2
57     4
58     6
59     2
60     3
61     2
62     3
63     4
64     1
65     3
66     4
Name: age, dtype: int64
```

### 5.8 class
```python
print('class:\n', diabetes_data['class'].value_counts(sort=False))
plotBar("Result (1 = positive, 0 = negative)", "Diabetes results", diabetes_data['class'])
```
<img src="/images/class_bar.png" title="Number of positives vs negatives" width="500" height="auto"/><br>
```
class:
0    456
1    217
Name: class, dtype: int64
```

### 5.9 Correlation heatmap
```python
correlation_heatmap(diabetes_data)
```
<img src="/images/heatmap.png" title="Pearson Correlation of Features" width="700" height="auto"/><br>

### 5.10 Pair plot
```python
sns.pairplot(diabetes_data, hue = 'class')
plt.show()
```
<img src="/images/pairplot.png" title="Pairplot of Features" width="auto" height="auto"/><br>


### 5.11 Pivot Table
```python
pivot_table1 = pd.pivot_table(diabetes_data, index = 'class', values = ['preg', 'plas', 'pres', 'skin'])
print(pivot_table1)

pivot_table2 = pd.pivot_table(diabetes_data, index = 'class', values = ['test', 'mass', 'pedi', 'age'])
print(pivot_table2)
```
```
             plas      preg       pres       skin
class                                            
0      109.313596  3.298246  68.945175  19.815789
1      140.622120  4.838710  70.838710  19.843318

             age       mass      pedi       test
class                                           
0      30.789474  30.775439  0.398202  58.660088
1      36.755760  34.763134  0.490309  72.483871
```

## 6) Feature Engineering

### 6.1 Exploration of new features
No new features created.

### 6.2 Split into Training and Testing Data
```python
# define x, y
X = diabetes_data.drop(['class'], axis = 1)
y = diabetes_data['class']

# split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
```

## 7) Model Building

### 7.1 Logistic Regression
```python
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr, X_train, y_train,cv=5)
print(cv)
print(cv.mean())
```
```
[0.7962963  0.81481481 0.81481481 0.75700935 0.72897196]
0.7823814468674282
```

### 7.2 Decision Tree
```python
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, X_train, y_train, cv=5)
print(cv)
print(cv.mean())
```
```
[0.66666667 0.73148148 0.72222222 0.58878505 0.65420561]
0.6726722049151955
```

### 7.3 Random Forest
```python
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf, X_train, y_train, cv=5)
print(cv)
print(cv.mean())
```
```
[0.81481481 0.83333333 0.75       0.73831776 0.74766355]
0.7768258913118726
```

## 8) Hyperparameter Tuning

### 8.1 Logistic Regression
* `C` : float, (default=1.0). Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

```python
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train, y_train)

print('Best Score: ' + str(best_clf_lr.best_score_))
print('Best Parameters: ' + str(best_clf_lr.best_params_))
```
```
Fitting 5 folds for each of 40 candidates, totalling 200 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=-1)]: Done  88 tasks      | elapsed:    1.3s
[Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed:    1.5s finished
Best Score: 0.7916753201799931
Best Parameters: {'C': 0.615848211066026, 'max_iter': 2000, 'penalty': 'l1', 'solver': 'liblinear'}
```
```python
y_predict = best_clf_lr.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Accuracy:", accuracy_score(y_test, y_predict))
```
```
Confusion Matrix:
 [[83  9]
 [20 23]]
Accuracy: 0.7851851851851852
```

### 8.2 Decision Tree
* `criterion` : optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.

* `max_depth` : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).

```python
gini_acc_scores = []
entropy_acc_scores = []

criterions = ["gini", "entropy"]

for criterion in criterions:
	for depth in range(25):
	    dt = tree.DecisionTreeClassifier(criterion=criterion, max_depth = depth+1, random_state=depth)
	    model = dt.fit(X_train,y_train)
	    
	    y_predict = dt.predict(X_test)

	    if criterion == "gini":
	    	gini_acc_scores.append(accuracy_score(y_test, y_predict))
	    else:
	    	entropy_acc_scores.append(accuracy_score(y_test, y_predict))
```
```python
figuresize = plt.figure(figsize=(12,8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
EntropyAcc = plt.plot(np.arange(25)+1, entropy_acc_scores, '--bo')   
GiniAcc = plt.plot(np.arange(25)+1, gini_acc_scores, '--ro')
legend = plt.legend(['Entropy', 'Gini'], loc ='lower right',  fontsize=15)
title = plt.title('Accuracy Score for Multiple Depths', fontsize=25)
xlab = plt.xlabel('Depth of Tree', fontsize=20)
ylab = plt.ylabel('Accuracy Score', fontsize=20)

plt.show()

print("Gini max accuracy:", max(gini_acc_scores))
print("Entropy max accuracy:", max(entropy_acc_scores))
```
<img src="/images/dt_accuracy_plot.png" title="Accuracy Score for Multiple Depths" width="600" height="auto"/><br>
```
Gini max accuracy: 0.762962962962963
Entropy max accuracy: 0.762962962962963
```
```python
dt = tree.DecisionTreeClassifier(max_depth = 1, random_state = 1)
dt = dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predict))
```
```
Accuracy: 0.762962962962963
```

### 8.3 Random Forest
* `max_depth` : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).
```python
acc_scores = []              
depth = np.arange(1, 30)

for i in depth:

    rf = RandomForestClassifier(n_estimators=25, max_depth=i, random_state=1)
    rf.fit(X_train,y_train)

    y_predict = rf.predict(X_test)

    acc_scores.append(accuracy_score(y_test, y_predict)) 
```
```python
figsize = plt.figure(figsize = (12,8))
plot = plt.plot(depth, acc_scores, 'r')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
xlab = plt.xlabel('Depth of the trees', fontsize = 20)
ylab = plt.ylabel('Accuracy', fontsize = 20)
title = plt.title('(Random Forest) Accuracy vs Depth of Trees', fontsize = 25)
plt.show()
```
<img src="/images/rf_accuracy_plot.png" title="(Random Forest) Accuracy vs Depth of Trees" width="600" height="auto"/><br>
```python
rf = RandomForestClassifier(n_estimators=25, max_depth=acc_scores.index(max(acc_scores))+1, random_state=1)
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predict))
```
```
Accuracy: 0.7851851851851852
```