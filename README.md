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
| Variable | Definition | Type | Key|
| :-------: | :---------:| :-------:| :-------:|
| preg | Pregnancies | Numerical |  |
| plas | Plasma Glucose | Numerical |  |
| pres | Blood Pressure | Numerical |  |
| skin | Skin Thickness | Numerical |  |
| test | Insulin Test | Numerical |  |
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
<img src="/images/preg_boxplot.png" title="Pregnancies box plot" width="400" height="auto"/><br>

#### 4.1.2 plas (Plasma Glucose)
```python
sns.boxplot(x=diabetes_data['plas'])
plt.show()
```
<img src="/images/plas_boxplot.png" title="Plasma Glucose box plot" width="400" height="auto"/><br>

#### 4.1.3 skin (Skin Thickness)
```python
sns.boxplot(x=diabetes_data['skin'])
plt.show()
```
<img src="/images/skin_boxplot.png" title="Skin Thickness box plot" width="400" height="auto"/><br>

#### 4.1.4 test (Insulin Test)
```python
sns.boxplot(x=diabetes_data['test'])
plt.show()
```
<img src="/images/test_boxplot.png" title="Insulin Test box plot" width="400" height="auto"/><br>

#### 4.1.5 mass (Body Mass Index)
```python
sns.boxplot(x=diabetes_data['mass'])
plt.show()
```
<img src="/images/mass_boxplot.png" title="Body Mass Index box plot" width="400" height="auto"/><br>

#### 4.1.6 pedi (Pedigree)
```python
sns.boxplot(x=diabetes_data['pedi'])
plt.show()
```
<img src="/images/pedi_boxplot.png" title="Pedigree box plot" width="400" height="auto"/><br>

#### 4.1.7 age
```python
sns.boxplot(x=diabetes_data['age'])
plt.show()
```
<img src="/images/age_boxplot.png" title="Age box plot" width="400" height="auto"/><br>


## 5) Data Exploration
This section explores the distribution of each variable.

### 5.1 preg (Pregnancies)

```python
print('preg (Pregnancies):\n', diabetes_data.preg.value_counts(sort=False))

# plot the distribution
plt.title("Histogram of number of entries per pregnancy count", fontsize=20)
plt.xlabel("Pregnancies", fontsize=16)  
plt.hist(diabetes_data.preg)
plt.show()
```
<img src="/images/preg_hist.png" title="Histogram of number of entries per pregnancy count" width="400" height="auto"/><br>
```
preg (Pregnancies):
 0     111
1     135
2     103
3      75
4      68
5      57
6      50
7      45
8      38
9      28
10     24
11     11
12      9
13     10
14      2
15      1
17      1
Name: preg, dtype: int64
```

### 5.2 plas (Plasma Glucose)
```python
print('plas (Plasma Glucose):\n', diabetes_data.plas.value_counts(sort=False))

# plot the distribution
plt.title("Histogram of number of entries per plasma glucose score", fontsize=20)
plt.xlabel("Plasma Glucose", fontsize=16)  
plt.hist(diabetes_data.plas)
plt.show()
```
<img src="/images/plas_hist.png" title="Histogram of number of entries per plasma glucose score" width="400" height="auto"/><br>
```
plas (Plasma Glucose):
 0      5
44     1
56     1
57     2
61     1
      ..
195    2
196    3
197    4
198    1
199    1
Name: plas, Length: 136, dtype: int64
```

### 5.3 skin
```python
print('skin (Skin Thickness):\n', diabetes_data.skin.value_counts(sort=False))

# plot the distribution
plt.title("Histogram of number of entries per skin thickness score", fontsize=20)
plt.xlabel("Skin Thickness", fontsize=16)  
plt.hist(diabetes_data.skin)
plt.show()
```
<img src="/images/skin_hist.png" title="Histogram of number of entries per skin thickness score" width="400" height="auto"/><br>
```
skin (Skin Thickness):
0     227
7       2
8       2
10      5
11      6
12      7
13     11
14      6
15     14
16      6
17     14
18     20
19     18
20     13
21     10
22     16
23     22
24     12
25     16
26     16
27     23
28     20
29     17
30     27
31     19
32     31
33     20
34      8
35     15
36     14
37     16
38      7
39     18
40     16
41     15
42     11
43      6
44      5
45      6
46      8
47      4
48      4
49      3
50      3
51      1
52      2
54      2
56      1
60      1
63      1
99      1
Name: skin, dtype: int64
```

### 5.4 mass
```python
print('mass (Body Mass Index):\n', diabetes_data.mass.value_counts(sort=False))

# plot the distribution
plt.title("Histogram of number of entries per body mass index score", fontsize=20)
plt.xlabel("Body Mass Index", fontsize=16)  
plt.hist(diabetes_data.mass)
plt.show()
```
<img src="/images/mass_hist.png" title="Histogram of number of entries per body mass index score" width="400" height="auto"/><br>
```
mass (Body Mass Index):
31.0     2
30.5     7
0.0     11
38.0     2
30.0     7
        ..
34.6     5
26.9     1
23.4     1
31.2    12
49.3     1
Name: mass, Length: 248, dtype: int64
```

### 5.5 pedi
```python
print('pedi (Pedigree):\n', diabetes_data.pedi.value_counts(sort=False))

# plot the distribution
plt.title("Histogram of number of entries per pedigree score", fontsize=20)
plt.xlabel("Pedigree", fontsize=16)  
plt.hist(diabetes_data.pedi)
plt.show()
```
<img src="/images/pedi_hist.png" title="Histogram of number of entries per pedigree score" width="400" height="auto"/><br>
```
pedi (Pedigree):
0.375    1
0.875    2
0.381    1
0.181    1
0.514    2
        ..
0.231    2
0.893    1
0.286    2
0.084    1
0.362    1
Name: pedi, Length: 517, dtype: int64
```

### 5.6 age
```python
print('age:\n', diabetes_data.age.value_counts(sort=False))

# plot the distribution
plt.title("Histogram of number of entries per age", fontsize=20)
plt.xlabel("Age", fontsize=16)  
plt.hist(diabetes_data.age)
plt.show()
```
<img src="/images/age_hist.png" title="Histogram of number of entries per age" width="400" height="auto"/><br>
```
age:
21    63
22    72
23    38
24    46
25    48
26    33
27    32
28    35
29    29
30    21
31    24
32    16
33    17
34    14
35    10
36    16
37    19
38    16
39    12
40    13
41    22
42    18
43    13
44     8
45    15
46    13
47     6
48     5
49     5
50     8
51     8
52     8
53     5
54     6
55     4
56     3
57     5
58     7
59     3
60     5
61     2
62     4
63     4
64     1
65     3
66     4
67     3
68     1
69     2
70     1
72     1
81     1
```

### 5.7 class
```python
print('class:\n', diabetes_data['class'].value_counts(sort=False))

# plot the distribution
plt.title("Number of positives vs negatives", fontsize=20) 
ax = sns.barplot(diabetes_data['class'].value_counts().index, diabetes_data['class'].value_counts())
ax.set(xlabel='Class', ylabel='# of entries')
plt.show()
```
<img src="/images/class_bar.png" title="Number of positives vs negatives" width="400" height="auto"/><br>
```
class:
0    500
1    268
Name: class, dtype: int64
```

### 5.8 Correlation heatmap
```python
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':10 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(diabetes_data)

plt.show()
```
<img src="/images/heatmap.png" title="Pearson Correlation of Features" width="700" height="auto"/><br>

### 5.9 Pair plot
```python
sns.pairplot(diabetes_data, hue = 'class')
plt.show()
```
<img src="/images/pairplot.png" title="Pairplot of Features" width="auto" height="auto"/><br>


### 5.10 Pivot Table
```python
pivot_table1 = pd.pivot_table(diabetes_data, index = 'class', values = ['preg', 'plas', 'pres', 'skin'])
print(pivot_table1)

pivot_table2 = pd.pivot_table(diabetes_data, index = 'class', values = ['test', 'mass', 'pedi', 'age'])
print(pivot_table2)
```
```
             plas      preg       pres       skin
class                                            
0      109.980000  3.298000  68.184000  19.664000
1      141.257463  4.865672  70.824627  22.164179


             age       mass      pedi        test
class                                            
0      31.190000  30.304200  0.429734   68.792000
1      37.067164  35.142537  0.550500  100.335821
```

## 6) Feature Engineering

### 6.1 Exploration of new features


### 6.2 Split into Training and Testing Data


## 7) Evaluate Model Performance

### 7.1 Data Preprocessing for Model


### 7.2 Model Building