# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:32:04 2024

@author: evenk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\DS_NIT\Machine Learning\Classifications\logit classification.csv")

x=data.iloc[:,[2,3]].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=21)

from sklearn.linear_model import LogisticRegression
cs=LogisticRegression(penalty='l2',dual=False,tol=1e-4,C=1.0,fit_intercept=True,intercept_scaling=1,
                      solver='lbfgs',max_iter=100,multi_class='auto')
cs.fit(x_train,y_train)

#predict the target value for the test
y_pred=cs.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

bias=cs.score(x_train,y_train)
print(bias)

variance=cs.score(x_test,y_test)
print(variance)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import classification_report
cf=classification_report(y_test,y_pred)
print(cf)

#Normalization
from sklearn.preprocessing import Normalizer
sc_x=Normalizer()
x_train=np.array(x_train).reshape(-1,1)
x_test=np.array(x_test).reshape(-1,1)

x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

#Standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_std=scaler.fit_transform(x_train)
x_test_std=scaler.fit_transform(x_test)





