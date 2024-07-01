import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("./dataset/slc.csv")

# print(data.dtypes)

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

# print("Top 5 Correlated features with lung cancer:")
# print(correlation_matrix['LUNG_CANCER'].sort_values(ascending=False).head())

# print("/nTop 5 important features:")
# print(feature_importance.head())

# print("/nCommon characteristics among Lung Cancer Patients:")
# print(common_characteristics)


#feature importances
plt.figure(figsize=(12,6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title('Top 10 important features for lung cancer prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#common characteristics
plt.figure(figsize=(12,6))
plt.bar(common_characteristics.index[:10], common_characteristics.values[:10])
plt.title('Top 10 common characteristics for lung cancer predicition')
plt.xlabel('Characteristics')
plt.ylabel('average value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()