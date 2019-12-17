# importing the necessary libraries
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import src.utilities as utils
import seaborn as sns

# setting display properties of pandas
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

# loading the data set
dataset = pd.read_csv('../data/accidents_2012_to_2014.csv')

#------------------------------------Handling the missing values--------------------------------------------------------------------
print("Shape before dropping null values: ", dataset.shape)
cleaned_data = dataset.dropna()
print(cleaned_data.isnull().sum()) #Number of null values in each column
print("Shape after dropping null values: ", cleaned_data.shape)
print(cleaned_data.columns)
print(cleaned_data.info())

#Mapping dates into Months
print(cleaned_data['Date'])
dataset['Date'] = pd.to_datetime(dataset['Date']).dt.month
dataset.rename(columns={'Date': 'Month'}, inplace=True)
print(dataset)
utils.write_file(dataset, '../data/date_and_time_converted.csv')