import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# read data set
diabetes_data = pd.read_csv("Data set/diabetes.csv", encoding= 'unicode_escape')


# ****************************************** Helper methods *******************************************

# helper methods
def remove_outliers_z_score(df):
	threshold = 3
	dataf = pd.DataFrame(df)
	z_scores = np.abs(stats.zscore(dataf))

	print("z-score outliers:", np.where(z_scores > threshold))
	print("# of outliers:", len(np.where(z_scores > threshold)[0]))
	
	return dataf[(z_scores  < 3).all(axis=1)]


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


# ****************************************** Remove Outliers *******************************************

#** PREG **#
sns.boxplot(x=diabetes_data['preg'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['preg'] = remove_outliers_iqr(diabetes_data['preg'])


#** PLAS **#
sns.boxplot(x=diabetes_data['plas'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['plas'] = remove_outliers_iqr(diabetes_data['plas'])


# #** SKIN **#
sns.boxplot(x=diabetes_data['skin'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['skin'] = remove_outliers_iqr(diabetes_data['skin'])


# #** TEST **#
sns.boxplot(x=diabetes_data['test'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['test'] = remove_outliers_iqr(diabetes_data['test'])


# #** MASS **#
sns.boxplot(x=diabetes_data['mass'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['mass'] = remove_outliers_iqr(diabetes_data['mass'])


# #** PEDI **#
sns.boxplot(x=diabetes_data['pedi'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['pedi'] = remove_outliers_iqr(diabetes_data['pedi'])


# #** AGE **#
sns.boxplot(x=diabetes_data['age'])
plt.show()

# calculate IQR score and remove outliers
diabetes_data['age'] = remove_outliers_iqr(diabetes_data['age'])


# ***************************************** Drop null data *****************************************

# drop all null rows
diabetes_data = diabetes_data.dropna()

print(diabetes_data.info())


# ***************************************  Convert object data type **************************************

diabetes_data['preg'] = diabetes_data['preg'].astype(int)
diabetes_data['plas'] = diabetes_data['plas'].astype(int)
diabetes_data['pres'] = diabetes_data['pres'].astype(int)
diabetes_data['skin'] = diabetes_data['skin'].astype(int)
diabetes_data['test'] = diabetes_data['test'].astype(int)
diabetes_data['age'] = diabetes_data['age'].astype(int)
diabetes_data['class'] = diabetes_data['class'].astype(int)

diabetes_data['mass'] = diabetes_data['mass'].astype(float)
diabetes_data['pedi'] = diabetes_data['pedi'].astype(float)


# ***************************************** Output to CSV *******************************************

diabetes_data.to_csv('Data set/diabetes_data_cleaned.csv',index = False)