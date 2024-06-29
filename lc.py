import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("./dataset/slc.csv")

print(data.dtypes)

for column in data.columns: 
    if data[column].dtypes == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    else:
        data[column].map({1:0, 2:1})


correlation_matrix = data.corr()
#features
X = data.drop('LUNG_CANCER', axis=1)
#target variable
y = data['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)