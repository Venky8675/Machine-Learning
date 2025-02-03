import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\DS_NIT\Machine Learning\Regression\MLR\Investment.csv")

x=dataset.iloc[:,:-1]

y=dataset.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred=regressor.predict(x_test)

m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)

#Backward elimination
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]
#Ordinary least squares(OLS)

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())

x_opt=x[:,[0,1,2,3,5]]
#Ordinary least squares(OLS)

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())

x_opt=x[:,[0,1,2,3,]]
#Ordinary least squares(OLS)

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())

x_opt=x[:,[0,1,3]]
#Ordinary least squares(OLS)

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())

x_opt=x[:,[0,1]]
#Ordinary least squares(OLS)

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())
