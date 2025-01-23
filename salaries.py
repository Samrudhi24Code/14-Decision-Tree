# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:27:27 2024

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r"E:\Data Science\14-Decision Tree\salaries.csv")

# Data preparation
# Check for null values and drop rows with missing data
print("Null values before dropping:", data.isnull().sum())
data = data.dropna()  # Drop rows with any null values
print("Null values after dropping:", data.isnull().sum())

# Initialize LabelEncoder
lb = LabelEncoder()

# Encode categorical columns
data["company"] = lb.fit_transform(data["company"])
data["job"] = lb.fit_transform(data["job"])
data["degree"] = lb.fit_transform(data["degree"])
data["salary_more_than_100k"] = lb.fit_transform(data["salary_more_than_100k"])

# Check for remaining non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# Split the columns into predictors and target
colnames = list(data.columns)
predictors = colnames[:-1]  # All columns except the last one
target = colnames[-1]  # The last column is the target

# Splitting data into training and testing datasets
#spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion='entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds_test = model.predict(test[predictors])
confusion_matrix_test = pd.crosstab(test[target], preds_test, rownames=['Actual'], colnames=['Prediction'])
print("Confusion Matrix on Test Data:\n", confusion_matrix_test)
accuracy_test = np.mean(preds_test == test[target])
print(f'Test Data Accuracy: {accuracy_test:.2f}')

# Prediction on Train Data
preds_train = model.predict(train[predictors])
confusion_matrix_train = pd.crosstab(train[target], preds_train, rownames=['Actual'], colnames=['Prediction'])
print("Confusion Matrix on Train Data:\n", confusion_matrix_train)
accuracy_train = np.mean(preds_train == train[target])
print(f'Train Data Accuracy: {accuracy_train:.2f}')




mode;.predict([[2,1,0]])
