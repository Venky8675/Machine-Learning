# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:31:53 2024

@author: evenk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\DS_NIT\Machine Learning\emp_sal.csv")

x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

#svm model
from sklearn.svm import SVR
svr_regressor=SVR(kernel='poly',degree=4,gamma='auto')
svr_regressor.fit(x,y)

svr_model_pred=svr_regressor.predict([[6.5]])
svr_model_pred

from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=4, weights='uniform',p=2)
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
knn_reg_pred

