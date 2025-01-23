# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:38:49 2024

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Business Objective:
# The goal is to build a Decision Tree model to predict credit default risk for customers based on their financial and personal details.

# Read the dataset
data = pd.read_csv(r"E:\Data Science\14-Decision Tree\credit.csv")

# Data Preparation:
# Check for null values and drop unnecessary columns
data.isnull().sum()  # Check for missing values
data.dropna()  # Drop any rows with missing data (if applicable)

# Drop the 'phone' column as it is irrelevant to the analysis
data = data.drop(["phone"], axis=1)

# Note: The dataset now has 16 columns.

# Encode non-numeric columns using LabelEncoder
lb = LabelEncoder()

# Transform categorical variables into numeric form for model compatibility
data["Checking_balance"] = lb.fit_transform(data["checking_balance"])
data["credit_history"] = lb.fit_transform(data["credit_history"])
data["purpose"] = lb.fit_transform(data["purpose"])
data["savings_balance"] = lb.fit_transform(data["savings_balance"])
data["employment_duration"] = lb.fit_transform(data["employment_duration"])
data["other_credit"] = lb.fit_transform(data["other_credit"])
data["housing"] = lb.fit_transform(data["housing"])
data["job"] = lb.fit_transform(data["job"])
data["default"] = lb.fit_transform(data["default"])

# Check for any remaining non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object']).columns
print(non_numeric_cols)  # Ensures all columns are numeric now

# Analyze the target column ('default') to understand the distribution
data["default"].unique()  # Unique values in target column
data["default"].value_counts()  # Count of each category in target column

# Define predictors (independent variables) and target (dependent variable)
colnames = list(data.columns)
predictors = colnames[:15]  # All columns except the last (target)
target = colnames[15]       # 'default' column is the target variable

# Split the data into training and testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)

# Build the Decision Tree Model:
# Using 'entropy' as the criterion to measure information gain
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion='entropy')  # Specify the decision tree type
model.fit(train[predictors], train[target])  # Train the model on training data

# Make Predictions on Test Data
preds = model.predict(test[predictors])  # Predict using test dataset

# Evaluate Model Accuracy on Test Data
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Prediction'])  # Confusion matrix
print("Test Data Accuracy:", np.mean(preds == test[target]))  # Calculate test accuracy

# Make Predictions on Training Data (for overfitting check)
preds = model.predict(train[predictors])  # Predict using training dataset

# Evaluate Model Accuracy on Training Data
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['Prediction'])  # Confusion matrix
print("Train Data Accuracy:", np.mean(preds == train[target]))  # Calculate train accuracy
