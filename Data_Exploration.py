import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# read data set
diabetes_data = pd.read_csv("Data set/diabetes_data_cleaned.csv", encoding= 'unicode_escape')


# ******************************************* Helper methos *******************************************

def plotHist(xlabel, title, column):

    # Remove the plot frame lines. They are unnecessary chartjunk.  
    fig, ax = plt.subplots(1, 1, 
                           figsize =(10, 7),  
                           tight_layout = True)

    # hide top and right spines for aesthetics
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)

    # Make sure axis ticks are large enough to be easily read.
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)

    # set labels and titles
    plt.xlabel(xlabel, fontsize=16)  
    plt.ylabel("# of entries", fontsize=16)
    plt.title(title, fontsize=20)

    plt.hist(column)
    plt.show()


def plotBar(xlabel, title, column):

    ax = sns.barplot(column.value_counts().index, column.value_counts())

    # hide top and right spines for aesthetics
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)

    # Make sure axis ticks are large enough to be easily read.
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)

    # set labels and titles
    plt.xlabel(xlabel, fontsize=16)  
    plt.ylabel("# of entries", fontsize=16)
    plt.title(title, fontsize=20)

    plt.show()


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


# ******************************************* Visualizations *******************************************

#** PREG **#
print('preg (Pregnancies):\n', diabetes_data.preg.value_counts(sort=False))
plotHist("Pregnancies", "Histogram of number of entries per number of pregnancies", diabetes_data.preg) 


#** PLAS **#
print('plas (Plasma Glucose):\n', diabetes_data.plas.value_counts(sort=False))
plotHist("Plasma Glucose Levels (mg/dL)", "Histogram of number of entries per plasma glucose levels", diabetes_data.plas) 


#** SKIN **#
print('skin (Skin Thickness):\n', diabetes_data.skin.value_counts(sort=False))
plotHist("Skin Thickness", "Histogram of number of entries per skin thickness length", diabetes_data.skin) 


#** TEST **#
print('test (Insulin Level):\n', diabetes_data.test.value_counts(sort=False))
plotHist("Insulin Level", "Histogram of number of entries per insulin level", diabetes_data.test) 


#** MASS **#
print('mass (Body Mass Index):\n', diabetes_data.mass.value_counts(sort=False))
plotHist("Body Mass Index", "Histogram of number of entries per body mass index score", diabetes_data.mass) 


#** PEDI **#
print('pedi (Pedigree):\n', diabetes_data.pedi.value_counts(sort=False))
plotHist("Pedigree", "Histogram of number of entries per pedigree count", diabetes_data.mass) 


#** AGE **#
print('age:\n', diabetes_data.age.value_counts(sort=False))
plotHist("Age", "Histogram of number of entries per age", diabetes_data.age) 


#** CLASS **#
print('class:\n', diabetes_data['class'].value_counts(sort=False))
plotBar("Result (0 = negative, 1 = positive)", "Diabetes results", diabetes_data['class'])


# correlation heatmap
correlation_heatmap(diabetes_data)


# pair plot
sns.pairplot(diabetes_data, hue = 'class')
plt.show()


# pivot table
pivot_table1 = pd.pivot_table(diabetes_data, index = 'class', values = ['preg', 'plas', 'pres', 'skin'])
print(pivot_table1)

pivot_table2 = pd.pivot_table(diabetes_data, index = 'class', values = ['test', 'mass', 'pedi', 'age'])
print(pivot_table2)