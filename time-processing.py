# importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import src.utilities as utils
import seaborn as sns
from datetime import datetime as dt, timedelta

# setting display properties of pandas
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

# loading the data set
data = pd.read_csv('../data/accidents_2012_to_2014.csv')
failed_rows = []
def get_time(time_string_series):
    for i in range(0, len(time_string_series)):
        #print(type(time_string_series[i]))
        if(isinstance(time_string_series[i], str) is False or not time_string_series[i]):
            failed_rows.append(i)
            continue;
        if(i%1000==0):
            print('i = ',i)
            utils.write_file(data, '../data/time_converted_'+str(i)+'.csv')
        time = dt.strptime(time_string_series[i], '%H:%M')
        if dt.strptime('07:00', "%H:%M") <= time and time < dt.strptime('11:00', "%H:%M"):
            time_string_series[i] = 'Morning'
        elif dt.strptime('11:00', "%H:%M") <= time and time < dt.strptime('15:00', "%H:%M"):
            time_string_series[i] = 'Noon'
        elif dt.strptime('15:00', "%H:%M") <= time and time < dt.strptime('19:00', "%H:%M"):
            time_string_series[i] = 'Afternoon'
        elif dt.strptime('19:00', "%H:%M") <= time and time <= dt.strptime('23:59', "%H:%M"):
            time_string_series[i] = 'Night'
        elif dt.strptime('00:00', "%H:%M") <= time and time < dt.strptime('07:00', "%H:%M"):
            time_string_series[i] = 'Late_night'
get_time(data['Time'])
print(data)
utils.write_file(data, '../data/accident-data-task2.csv')
print("Failed Rows: ", failed_rows)
