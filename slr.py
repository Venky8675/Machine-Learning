
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\DS_NIT\Machine Learning\Salary_Data.csv")

x = dataset.iloc[:, :-1]  # Years of experience (Independent variable)
y = dataset.iloc[:, -1]   # Salary (Dependent variable)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Visualizing the Test set results
plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


print(regressor)

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_slope * 12 + c_intercept
print(y_12)









