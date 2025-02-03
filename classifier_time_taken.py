# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:00:48 2024

@author: evenk
"""

from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)

clf=LazyClassifier(verbose=0,ignore_warnings=True,custom_metric=None)

models,predictions=clf.fit(x_train,x_test,y_train,y_test)

print(models)
