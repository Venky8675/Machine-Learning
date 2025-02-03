import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\DS_NIT\Machine Learning\emp_sal.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#linear model-- linear algor(degree-1)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
lin_reg.coef_
lin_reg.intercept_
lin_reg.predict(x)


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


#Linear regression visualization
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear Regression graph')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Polynomial visualization
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or Bluff (Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#prediction
lin_model_pred=lin_reg.predict([[6.5]])
lin_model_pred


poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred