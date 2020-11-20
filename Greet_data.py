import pandas as pd

# read data set
diabetes_data = pd.read_csv("Data set/diabetes.csv", encoding= 'unicode_escape')

# get a peek at the top 5 rows of the data set
print(diabetes_data.head())

# understand the type of each column
print(diabetes_data.info())

# get information on the numerical columns for the data set
with pd.option_context('display.max_columns', len(diabetes_data.columns)):
    print(diabetes_data.describe(include='all'))