
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\DS_NIT\Machine Learning\emp_sal.csv")

x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

#svr model
from sklearn.svm import SVR
svr_regressor=SVR(kernel='poly',degree=4,gamma='scale')
svr_regressor.fit(x,y)

svr_model_pred=svr_regressor.predict([[6.5]])
svr_model_pred

#KNN model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=5, weights='uniform',algorithm='auto',p=2)
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
knn_reg_pred


#decision tree
from sklearn.tree import DecisionTreeRegressor

# Creating and fitting the regressor
dt_regressor = DecisionTreeRegressor(criterion='poisson',splitter='best',min_samples_split=3,max_depth=4,max_features='log2')
dt_regressor.fit(x, y)

# Making predictions
dt_reg_pred = dt_regressor.predict([[6.5]])
dt_reg_pred


#Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_regressor=RandomForestRegressor(n_estimators=4,criterion='absolute_error',max_depth=3,min_samples_split=4,random_state=1)
rf_regressor.fit(x,y)

rf_reg_pred=rf_regressor.predict([[6.5]])
rf_reg_pred



