import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"D:\DS_NIT\Machine Learning\Classifications\insurance_data.csv")

# Prepare the data
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, -1].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train the Logistic Regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy Score:", ac)

# Use the model's score method for bias and variance
bias = model.score(x_train, y_train)
print(bias)

variance = model.score(x_test, y_test)
print(variance)

#Feature prediction
data1=pd.read_excel(r"D:\DS_NIT\Machine Learning\Classifications\insurance_future.xlsx")
print(data1)

d2=data1.copy()

data1=data.iloc[:,1].values.reshape(-1, 1)

M=sc.fit_transform(data1)
y_pred1=pd.DataFrame()

d2['y_pred1']=model.predict(M)
print(d2)

d2.to_csv('pred_model.csv')

import os
os.getcwd()



