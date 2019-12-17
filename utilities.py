import datetime as dt

#this function save as dataframe as a csv file in the specified path
def write_file(dataframe, filePath):
    dataframe.to_csv(filePath, index=False)

#this function is used to convert the times into five time categories
def get_time(time_string_series):
    failed_rows = []
    for i in range(0, len(time_string_series)):
        # if there is any misformed row, skip it and save the row number
        if(isinstance(time_string_series[i], str) is False or not time_string_series[i]):
            failed_rows.append(i)
            continue;
        #convert the time string to time object
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
    return failed_rows