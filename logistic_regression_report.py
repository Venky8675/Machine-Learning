import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"D:\DS_NIT\Machine Learning\Classifications\logit classification.csv")

x=data.iloc[:,[2,3]].values

y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
cs=LogisticRegression()
cs.fit(x_train,y_train)
y_pred=cs.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

bias=cs.score(x_train,y_train)
print(bias)

variance=cs.score(x_test,y_test)
print(variance)



#Feautre prediction
data1=pd.read_csv(r"D:\DS_NIT\Dec'24\15. Logistic regression with future prediction\final1.csv")

d2=data1.copy()
data1=data1.iloc[:,[3,4]].values

M=sc.fit_transform(data1)

y_pred1=pd.DataFrame()

d2['y_pred1']=cs.predict(M)

print(d2)

d2.to_csv("pred_model_new.csv")

import os
os.getcwd()


from sklearn.metrics import classification_report

print('Classification Report:')
print(classification_report(y_test,y_pred))

from sklearn.model_selection import cross_val_score
cv_score=cross_val_score(cs,x,y,cv=5,scoring='accuracy')
print('Cross-Validation scores:',cv_score)
print('Mean accuracy from cross-validation:',cv_score.mean())
