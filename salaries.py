# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:27:27 2024

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Business Objective: 
# The goal is to build a Decision Tree model to predict whether an individual earns more than $100k based on various factors like company, job, and degree.
# This is a classification problem where the target variable is binary: 'salary_more_than_100k' (Yes/No).

# Load the dataset
data = pd.read_csv(r"E:\Data Science\14-Decision Tree\salaries.csv")

# Data preparation
# Check for null values and drop rows with missing data
print("Null values before dropping:", data.isnull().sum())
data = data.dropna()  # Drop rows with any null values (no missing values in the dataset anymore)
print("Null values after dropping:", data.isnull().sum())

# Initialize LabelEncoder to encode categorical data into numerical format
lb = LabelEncoder()

# Encode categorical columns
# These columns have string values that need to be converted into numeric values for the model to process.
data["company"] = lb.fit_transform(data["company"])  # Encode company names to numeric labels
data["job"] = lb.fit_transform(data["job"])  # Encode job titles to numeric labels
data["degree"] = lb.fit_transform(data["degree"])  # Encode degree types to numeric labels
data["salary_more_than_100k"] = lb.fit_transform(data["salary_more_than_100k"])  # Target encoding for salary

# Check for remaining non-numeric columns after encoding
non_numeric_cols = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# Split the columns into predictors (independent variables) and target (dependent variable)
colnames = list(data.columns)
predictors = colnames[:-1]  # All columns except the last one (salary_more_than_100k)
target = colnames[-1]  # The last column is the target variable (salary_more_than_100k)

# Splitting data into training and testing datasets
from sklearn.model_selection import train_test_split
# 70% for training and 30% for testing
train, test = train_test_split(data, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion='entropy')  # Using 'entropy' as the criterion for splitting
model.fit(train[predictors], train[target])  # Train the model on the training data

# Prediction on Test Data
# Make predictions on the test data to evaluate model performance
preds_test = model.predict(test[predictors])
# Display confusion matrix for test data
confusion_matrix_test = pd.crosstab(test[target], preds_test, rownames=['Actual'], colnames=['Prediction'])
print("Confusion Matrix on Test Data:\n", confusion_matrix_test)
accuracy_test = np.mean(preds_test == test[target])  # Calculate test data accuracy
print(f'Test Data Accuracy: {accuracy_test:.2f}')

# Prediction on Train Data
# Make predictions on the training data to check for overfitting
preds_train = model.predict(train[predictors])
# Display confusion matrix for train data
confusion_matrix_train = pd.crosstab(train[target], preds_train, rownames=['Actual'], colnames=['Prediction'])
print("Confusion Matrix on Train Data:\n", confusion_matrix_train)
accuracy_train = np.mean(preds_train == train[target])  # Calculate train data accuracy
print(f'Train Data Accuracy: {accuracy_train:.2f}')

# Predict for a new sample (Example: company=2, job=1, degree=0)
# Business Solution: We use the trained model to predict whether an individual with these features earns more than $100k.
# This can be useful for companies to predict the salary bracket of potential employees based on their job role, company, and degree.
sample_data = [[2, 1, 0]]  # Replace with actual values based on your dataset's feature encoding
prediction = model.predict(sample_data)  # Predict for the given sample
print(f"Prediction for the sample data {sample_data}: {prediction[0]}")
# The prediction will be 0 (No) or 1 (Yes), indicating whether the predicted salary is more than $100k.

