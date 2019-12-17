import pandas as pd
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
import numpy as np
import warnings

# ignoring the warnings
warnings.filterwarnings('ignore')

# setting the seaborn chart type
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# setting display properties of pandas
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

# loading the data set
# in the dataset, the time is already converted to time categories
cleaned_data = pd.read_csv('../data/date_and_time_converted.csv', header=0)

# mapping fatal accident to true and other two categories to false and hence changing the attribute name
cleaned_data['Accident_Severity'] = cleaned_data['Accident_Severity'].map({1: 1, 2: 0, 3: 0})
cleaned_data.rename(columns={'Accident_Severity': 'Accident_Severity_Fatal'}, inplace=True)
# print(cleaned_data.head())

print("Shape of the Dataset: ", cleaned_data.shape)
print("Column of the dataset: ", cleaned_data.columns)

# dropping the irrelevant columns

reduced_dataset = pd.DataFrame(cleaned_data,
                               columns=['Month', 'Day_of_Week', 'Time', 'Road_Type', 'Speed_limit',
                                        'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities',
                                        'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions',
                                        'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Urban_or_Rural_Area',
                                        'Accident_Severity_Fatal'])

print("Shape of dataset after dropping irrelevant columns: ", reduced_dataset)
print("Counting missing values in each column: ")
print(reduced_dataset.isnull().sum())  # Number of null values in each column
print("Shape of data before dropping missing values: ", reduced_dataset.shape)
reduced_dataset = reduced_dataset.dropna()
print("Shape of data after dropping missing values: ", reduced_dataset.shape)
print("Counting missing values in each column after dropping:")
print(reduced_dataset.isnull().sum())  # Number of null values in each column
print("Count of Fatal/Non-fatal accidents: ", reduced_dataset['Accident_Severity_Fatal'].value_counts())

# show the count plot
# fig, ax1 = plt.subplots(figsize=(8,5))
# graph = sns.countplot(ax=ax1, x='Accident_Severity_Fatal', data=reduced_dataset, palette='hls')
# graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
# plt.title("Accident Severity Plot")
# for p in graph.patches:
#     height = p.get_height()
#     graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
# plt.show()

count_fatal = len(cleaned_data[cleaned_data['Accident_Severity_Fatal'] == 1])
count_non_fatal = len(cleaned_data[cleaned_data['Accident_Severity_Fatal'] == 0])
percentage_non_fatal = count_non_fatal / (count_fatal + count_non_fatal)
percentage_fatal = count_fatal / (count_fatal + count_non_fatal)
print("Percentage of fatal accident: ", percentage_fatal * 100)
print("Percentage of non-fatal accident: ", percentage_non_fatal * 100)

print("Columns in the reduced dataset: ", reduced_dataset.columns)

# Setting numerical Columns as cetgory
num_columns = ['Month', 'Day_of_Week', 'Speed_limit', 'Urban_or_Rural_Area', 'Accident_Severity_Fatal']
for col in num_columns:
    reduced_dataset[col] = reduced_dataset[col].astype('category')

# some basic statisticsa and analysis

# pd.crosstab(reduced_dataset.Month, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Month")
# plt.xlabel('Month')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Day_of_Week, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title('Accident Frequency for Days of Week')
# plt.xlabel('Days of Week')
# plt.ylabel('Frequency of Fatal Accident')
# plt.show()
#
# pd.crosstab(reduced_dataset.Time, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Time")
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Road_Type, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Road Type")
# plt.xlabel('Road Type')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Speed_limit, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title('Accident Frequency for Speed Limit')
# plt.xlabel('Speed Limit')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset['Pedestrian_Crossing-Human_Control'], reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Pedestrian Crossing Human Control")
# plt.xlabel('Pedestrian Crossing Human Control')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset['Pedestrian_Crossing-Physical_Facilities'], reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Pedestrian Crossing Physical Facilities")
# plt.xlabel('Pedestrian Crossing Physical Facilities')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Light_Conditions, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Light Condition")
# plt.xlabel('Light Condition')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Weather_Conditions, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Weather Condition")
# plt.xlabel('Weather Condition')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Road_Surface_Conditions, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Road Surface Condition")
# plt.xlabel('Road Surface Condition')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Special_Conditions_at_Site, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Special Condition at Site")
# plt.xlabel('Special Condition at Site')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Carriageway_Hazards, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Carriageway Hazard")
# plt.xlabel('Carriageway Hazard')
# plt.ylabel('Frequency')
# plt.show()
#
# pd.crosstab(reduced_dataset.Urban_or_Rural_Area, reduced_dataset.Accident_Severity_Fatal).plot(kind='bar')
# plt.title("Accident Frequency for Urban or Rural Area")
# plt.xlabel('Urban or Rural Area')
# plt.ylabel('Frequency')
# plt.show()

# ---------------------------------------------Creating dummy variables-------------------------------------------------

categorical_vars = ['Month', 'Day_of_Week', 'Time', 'Road_Type', 'Speed_limit', 'Pedestrian_Crossing-Human_Control',
                    'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions', 'Weather_Conditions',
                    'Road_Surface_Conditions', 'Special_Conditions_at_Site', 'Carriageway_Hazards',
                    'Urban_or_Rural_Area']
for var in categorical_vars:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(reduced_dataset[var], prefix=var)
    data_temp = reduced_dataset.join(cat_list)
    reduced_dataset = data_temp

data_vars = reduced_dataset.columns.values.tolist()
to_keep = [i for i in data_vars if i not in categorical_vars]
data_final = reduced_dataset[to_keep]
print("Columns in final data: ", data_final.columns)

# ---------------------------------------------------------Splting data and Solving class imbalance problem---------------------------
X = data_final.loc[:, data_final.columns != 'Accident_Severity_Fatal']
y = data_final.loc[:, data_final.columns == 'Accident_Severity_Fatal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os = SMOTE(random_state=0)
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['Accident_Severity_Fatal'])

#numbers of our data
print("Length of oversampled data is ", len(os_data_X))
print("Number of non-fatal accident in oversampled data", len(os_data_y[os_data_y['Accident_Severity_Fatal'] == 0]))
print("Number of fatal accident", len(os_data_y[os_data_y['Accident_Severity_Fatal'] == 1]))
print("Proportion of non-fatal data in oversampled data is ",
      len(os_data_y[os_data_y['Accident_Severity_Fatal'] == 0]) / len(os_data_X))
print("Proportion of subscription data in oversampled data is ",
      len(os_data_y[os_data_y['Accident_Severity_Fatal'] == 1]) / len(os_data_X))

#------------------------------------------Recursive feature selection with cross validation------------------------------------------------------------

#from sklearn.model_selection import StratifiedKFold
#from sklearn.feature_selection import RFECV
#logreg = LogisticRegression()
# # from sklearn.svm import SVC
# # svc = SVC(kernel='linear')
# # Create the RFE object and compute a cross-validated score.
# rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(2),
#               scoring='accuracy')
# rfecv.fit(os_data_X, os_data_y.values.ravel())
# print("Optimal number of features : %d" % rfecv.n_features_)
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

# ------------------------------------------Recursive feature elimination------------------------------------------------
# data_final_vars = X.columns.values.tolist()
#
# y = ['Accident_Severity_Fatal']
# X = [i for i in data_final_vars if i not in y]
#
# logreg = LogisticRegression()
# rfe = RFE(logreg, 50)
# rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)
# print("Type of ", type(rfe.support_))

# # These columns are selected after recursive feature elimination
# X = []
# for i in range(0, len(rfe.support_)):
#     if (rfe.support_[i] == True):
#         X.append(data_final_vars[i])
# print("Selected Columns: ", X)
#-------------------------------------------------------Different number of features selected by the RFE---------------------------------------------
#top 10 features
# cols = ['Month_2', 'Month_12', 'Road_Type_Dual carriageway', 'Road_Type_Single carriageway', 'Speed_limit_60',
#         'Speed_limit_70', 'Weather_Conditions_Snowing with high winds', 'Special_Conditions_at_Site_None',
#         'Carriageway_Hazards_None', 'Carriageway_Hazards_Pedestrian in carriageway (not injured)']

# top 25 with features
# cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#  'Month_11', 'Month_12', 'Day_of_Week_1', 'Day_of_Week_7', 'Road_Type_Dual carriageway', 'Road_Type_Single carriageway',
#  'Speed_limit_60', 'Speed_limit_70', 'Pedestrian_Crossing-Human_Control_Control by school crossing patrol',
#  'Weather_Conditions_Fine without high winds', 'Weather_Conditions_Snowing with high winds',
#  'Special_Conditions_at_Site_None', 'Carriageway_Hazards_Dislodged vehicle load in carriageway',
#  'Carriageway_Hazards_None', 'Carriageway_Hazards_Pedestrian in carriageway (not injured)']
#

# #top 30 features
# cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#         'Month_11', 'Month_12', 'Day_of_Week_1', 'Day_of_Week_2', 'Day_of_Week_3', 'Day_of_Week_4', 'Day_of_Week_5',
#         'Day_of_Week_6', 'Day_of_Week_7', 'Road_Type_Dual carriageway', 'Road_Type_Single carriageway',
#         'Speed_limit_60', 'Speed_limit_70', 'Pedestrian_Crossing-Human_Control_Control by school crossing patrol',
#         'Weather_Conditions_Fine without high winds', 'Weather_Conditions_Snowing with high winds',
#         'Special_Conditions_at_Site_None', 'Carriageway_Hazards_Dislodged vehicle load in carriageway',
#         'Carriageway_Hazards_None', 'Carriageway_Hazards_Pedestrian in carriageway (not injured)']

# top 35 features
# cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#         'Month_11', 'Month_12', 'Day_of_Week_1', 'Day_of_Week_2', 'Day_of_Week_3', 'Day_of_Week_4', 'Day_of_Week_5',
#         'Day_of_Week_6', 'Day_of_Week_7', 'Time_Afternoon', 'Time_Late_night', 'Time_Morning', 'Time_Night',
#         'Time_Noon', 'Road_Type_Dual carriageway', 'Road_Type_Single carriageway', 'Speed_limit_60', 'Speed_limit_70',
#         'Pedestrian_Crossing-Human_Control_Control by school crossing patrol',
#         'Weather_Conditions_Fine without high winds', 'Weather_Conditions_Snowing with high winds',
#         'Special_Conditions_at_Site_None', 'Carriageway_Hazards_Dislodged vehicle load in carriageway',
#         'Carriageway_Hazards_None', 'Carriageway_Hazards_Pedestrian in carriageway (not injured)']

# top 45 features
# cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#         'Month_11', 'Month_12', 'Day_of_Week_1', 'Day_of_Week_2', 'Day_of_Week_3', 'Day_of_Week_4', 'Day_of_Week_5',
#         'Day_of_Week_6', 'Day_of_Week_7', 'Time_Afternoon', 'Time_Late_night', 'Time_Morning', 'Time_Night',
#         'Time_Noon', 'Road_Type_Dual carriageway', 'Road_Type_One way street', 'Road_Type_Single carriageway',
#         'Speed_limit_60', 'Speed_limit_70', 'Pedestrian_Crossing-Human_Control_Control by other authorised person',
#         'Pedestrian_Crossing-Human_Control_Control by school crossing patrol',
#         'Pedestrian_Crossing-Physical_Facilities_Central refuge',
#         'Pedestrian_Crossing-Physical_Facilities_Footbridge or subway',
#         'Pedestrian_Crossing-Physical_Facilities_No physical crossing within 50 meters',
#         'Pedestrian_Crossing-Physical_Facilities_Pedestrian phase at traffic signal junction',
#         'Pedestrian_Crossing-Physical_Facilities_non-junction pedestrian crossing',
#         'Weather_Conditions_Fine without high winds', 'Weather_Conditions_Snowing with high winds',
#         'Special_Conditions_at_Site_None', 'Special_Conditions_at_Site_Permanent sign or marking defective or obscured',
#         'Carriageway_Hazards_Dislodged vehicle load in carriageway', 'Carriageway_Hazards_None',
#         'Carriageway_Hazards_Pedestrian in carriageway (not injured)', 'Urban_or_Rural_Area_1', 'Urban_or_Rural_Area_2']

#top 55 featu
cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
        'Month_11', 'Month_12', 'Day_of_Week_1', 'Day_of_Week_2', 'Day_of_Week_3', 'Day_of_Week_4', 'Day_of_Week_5',
        'Day_of_Week_6', 'Day_of_Week_7', 'Time_Afternoon', 'Time_Late_night', 'Time_Morning', 'Time_Night',
        'Time_Noon', 'Road_Type_Dual carriageway', 'Road_Type_One way street', 'Road_Type_Single carriageway',
        'Speed_limit_30', 'Speed_limit_50', 'Speed_limit_60', 'Speed_limit_70',
        'Pedestrian_Crossing-Human_Control_Control by other authorised person',
        'Pedestrian_Crossing-Human_Control_Control by school crossing patrol',
        'Pedestrian_Crossing-Physical_Facilities_Central refuge',
        'Pedestrian_Crossing-Physical_Facilities_Footbridge or subway',
        'Pedestrian_Crossing-Physical_Facilities_No physical crossing within 50 meters',
        'Pedestrian_Crossing-Physical_Facilities_Pedestrian phase at traffic signal junction',
        'Pedestrian_Crossing-Physical_Facilities_Zebra crossing',
        'Pedestrian_Crossing-Physical_Facilities_non-junction pedestrian crossing',
        'Light_Conditions_Darkeness: No street lighting', 'Light_Conditions_Darkness: Street lights present and lit',
        'Light_Conditions_Daylight: Street light present', 'Weather_Conditions_Fine without high winds',
        'Weather_Conditions_Snowing with high winds', 'Road_Surface_Conditions_Dry', 'Road_Surface_Conditions_Wet/Damp',
        'Special_Conditions_at_Site_Auto traffic singal out', 'Special_Conditions_at_Site_None',
        'Special_Conditions_at_Site_Permanent sign or marking defective or obscured',
        'Special_Conditions_at_Site_Road surface defective',
        'Carriageway_Hazards_Dislodged vehicle load in carriageway', 'Carriageway_Hazards_None',
        'Carriageway_Hazards_Pedestrian in carriageway (not injured)', 'Urban_or_Rural_Area_1', 'Urban_or_Rural_Area_2']

X = os_data_X[cols]
y = os_data_y['Accident_Severity_Fatal']

# import statsmodels.api as sm
#
# logit_model = sm.Logit(y, X)
# result = logit_model.fit()
# print(result.summary2())

# building the logistic regression model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Printing the confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Precision recall summary
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
