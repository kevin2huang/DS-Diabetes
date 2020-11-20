import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# read data set
diabetes_data = pd.read_csv("Data set/diabetes.csv", encoding= 'unicode_escape')

# list of columns
# Index(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'], dtype='object')


# ******************************************* Graph Settings *******************************************

# Remove the plot frame lines. They are unnecessary chartjunk.  
fig, ax = plt.subplots(1, 1, 
                       figsize =(10, 7),  
                       tight_layout = True)

ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  

# Make sure axis ticks are large enough to be easily read.
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  

# y label
plt.ylabel("# of entries", fontsize=16)  


# ******************************************* Visualizations *******************************************

#** PREG **#
# print('preg (Pregnancies):\n', diabetes_data.preg.value_counts(sort=False))

# plot the distribution
# plt.title("Histogram of number of entries per pregnancy count", fontsize=20)
# plt.xlabel("Pregnancies", fontsize=16)  
# plt.hist(diabetes_data.preg)
# plt.show()



#** PLAS **#
# print('plas (Plasma Glucose):\n', diabetes_data.plas.value_counts(sort=False))

# # plot the distribution
# plt.title("Histogram of number of entries per plasma glucose score", fontsize=20)
# plt.xlabel("Plasma Glucose", fontsize=16)  
# plt.hist(diabetes_data.plas)
# plt.show()


#** SKIN **#
# print('skin (Skin Thickness):\n', diabetes_data.skin.value_counts(sort=False))

# # plot the distribution
# plt.title("Histogram of number of entries per skin thickness score", fontsize=20)
# plt.xlabel("Skin Thickness", fontsize=16)  
# plt.hist(diabetes_data.skin)
# plt.show()


#** TEST **#
# print('test (Insulin Test):\n', diabetes_data.test.value_counts(sort=False))

# plot the distribution
# plt.title("Histogram of number of entries per insulin test score", fontsize=20)
# plt.xlabel("Insulin Test", fontsize=16)  
# plt.hist(diabetes_data.test)
# plt.show()


#** MASS **#
# print('mass (Body Mass Index):\n', diabetes_data.mass.value_counts(sort=False))

# # plot the distribution
# plt.title("Histogram of number of entries per body mass index score", fontsize=20)
# plt.xlabel("Body Mass Index", fontsize=16)  
# plt.hist(diabetes_data.mass)
# plt.show()


#** PEDI **#
# print('pedi (Pedigree):\n', diabetes_data.pedi.value_counts(sort=False))

# # plot the distribution
# plt.title("Histogram of number of entries per pedigree score", fontsize=20)
# plt.xlabel("Pedigree", fontsize=16)  
# plt.hist(diabetes_data.pedi)
# plt.show()


#** AGE **#
# print('age:\n', diabetes_data.age.value_counts(sort=False))

# # plot the distribution
# plt.title("Histogram of number of entries per age", fontsize=20)
# plt.xlabel("Age", fontsize=16)  
# plt.hist(diabetes_data.age)
# plt.show()


# heatmap
# def correlation_heatmap(df):
#     _ , ax = plt.subplots(figsize =(14, 12))
#     colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
#     _ = sns.heatmap(
#         df.corr(), 
#         cmap = colormap,
#         square=True, 
#         cbar_kws={'shrink':.9 }, 
#         ax=ax,
#         annot=True, 
#         linewidths=0.1,vmax=1.0, linecolor='white',
#         annot_kws={'fontsize':10 }
#     )
    
#     plt.title('Pearson Correlation of Features', y=1.05, size=15)

# correlation_heatmap(diabetes_data)

# plt.show()


# pair plot
# sns.pairplot(diabetes_data, hue = 'class')
# plt.show()