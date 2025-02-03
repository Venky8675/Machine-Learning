import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r"D:\DS_NIT\Machine Learning\Classifications\logit classification.csv")
x=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#SVM
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

bias=svm.score(x_train,y_train)
variance=svm.score(x_test,y_test)
print(bias)
print(variance)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_cl=KNeighborsClassifier()
knn_cl.fit(x_train,y_train)

#predict knn outcomes
knn_y_pred=knn_cl.predict(x_test)

#KNN Confusion matrix 
knn_cm=confusion_matrix(y_test,knn_y_pred)
print('KNN_Confusion matrix:\n',knn_cm)

#KNN Classification Report
knn_cr=classification_report(y_test,knn_y_pred)
print('KNN_Classification report:\n',knn_cr)


#KNN_Accuracy
knn_ac=accuracy_score(y_test,knn_y_pred)
print('KNN accuracy\n',knn_ac)

#Logistic regresision
from sklearn.linear_model import LogisticRegression
log_cl=LogisticRegression()
#Train the logistic regression classifier
log_cl.fit(x_train,y_train)

logreg_ypred=log_cl.predict(x_test)

#Confusion matrix
log_cm=confusion_matrix(y_test,logreg_ypred)
print('Logistic Regresssion confusion matrix:\n',log_cm)

#classification report
log_cr=classification_report(y_test,logreg_ypred)
print('Logistic Regression classification report:\n',log_cr)

#Accuracy
log_ac=accuracy_score(y_test, logreg_ypred)
print('Logistic Regreesion Accuracy: ',log_ac)


#Validation testing
dataset1=pd.read_excel(r"D:\DS_NIT\Dec'24\15. Logistic regression with future prediction\final1.xlsx")
d2=dataset1.copy()

#Fetching dataset1
dataset1=dataset1.iloc[:,[3,4]].values

#Performing standard scaler
sc1=StandardScaler()
M=sc1.fit_transform(dataset1)

y_pred1=pd.DataFrame()

#Predict outcomes
d2['svm y_pred']=svm.predict(M)
d2['knn y_pred']=knn_cl.predict(M)
d2['Logistic y_pred']=log_cl.predict(M)

d2.to_csv('Pred model with knn_logreg,svm.csv',index=False)

import os
os.getcwd()









                    
