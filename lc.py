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

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_classifier.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

lung_cancer_patients = data[data['LUNG_CANCER'] == 1]
common_characteristics = lung_cancer_patients.mean().sort_values(ascending=False)