import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("./dataset/slc.csv")

print(data.dtypes)

for column in data.columns: 
    if data[column].dytypes == object:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    else:
        data[column].map({1:0, 2:1})
