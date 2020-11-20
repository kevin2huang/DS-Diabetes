import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# read data set
diabetes_data = pd.read_csv("Data set/diabetes.csv", encoding= 'unicode_escape')

# Index(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'], dtype='object')

# ****************************************** Remove Outliers *******************************************

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



#** PREG **#
sns.boxplot(x=diabetes_data['preg'])
plt.show()


#** PLAS **#
sns.boxplot(x=diabetes_data['plas'])
plt.show()


#** SKIN **#
sns.boxplot(x=diabetes_data['skin'])
plt.show()


#** TEST **#
sns.boxplot(x=diabetes_data['test'])
plt.show()


#** MASS **#
sns.boxplot(x=diabetes_data['mass'])
plt.show()


#** PEDI **#
sns.boxplot(x=diabetes_data['pedi'])
plt.show()


#** AGE **#
sns.boxplot(x=diabetes_data['age'])
plt.show()




# diabetes_data.to_csv('Data set/diabetes_data_cleaned.csv',index = False)