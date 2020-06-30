# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:25:24 2019

@author: 140524
"""

# K-Fold and cross validation

# example 1

from sklearn.model_selection import KFold
import numpy as np
data = np.arange(25)
Kf = KFold(5,False,random_state=5)
for train, test in kf.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))
    
#example 2: digit dataset in scikilearn

#https://www.youtube.com/watch?v=gJo0uNL-5Qw&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=13
#choosing the model
    



from sklearn.datasets import load_digits

digit = load_digits()
digit.data
digit.target

#score of logistic regression below is 0.96
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(digit.data,digit.target,test_size = 0.3)
logreg= LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
y_pred
logreg.score(X_test,y_test)

#score of svm is poor - 0.5

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)

# score of randomforest classifier is best at 0.97   
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 40)    
rf.fit(X_train,y_train)
rf.score(X_test,y_test)

#but distribution of response classes are not uniform in test train spolit

#using KFold with 3 folds

from sklearn.model_selection import KFold
kf = KFold(n_splits = 3,random_state = None, shuffle = False)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)
    
    
def get_score(model,X_train, X_test, y_train,y_test):
         model.fit(X_train,y_train)
         return model.score(X_test,y_test)
get_score(LogisticRegression(),X_train, X_test, y_train,y_test)
get_score(SVC(),X_train, X_test, y_train,y_test)
get_score(RandomForestClassifier(n_estimators = 40) ,X_train, X_test, y_train,y_test)

from sklearn.model_selection import StratifiedKFold
sf = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)
logregscore = []
svc_score = []
rf_score = []
from sklearn.datasets import load_digits

digit = load_digits()
digit.data
digit.target

for train_index, test_index in sf.split(digit.data,digit.target):
    X_train, X_test,y_train,y_test = digit.data[train_index],digit.data[test_index],\
                                     digit.target[train_index],digit.target[test_index]
    logregscore.append(get_score(LogisticRegression(),X_train, X_test, y_train,y_test))  
    svc_score.append(get_score(SVC(),X_train, X_test, y_train,y_test))
    rf_score.append(get_score(RandomForestClassifier(n_estimators = 40) ,X_train, X_test, y_train,y_test))

logregscore
svc_score
rf_score
                                
# the above steps can be replaced by using using the cross validation score

from sklearn.model_selection import cross_val_score
logreg=cross_val_score(LogisticRegression(),digit.data,digit.target, cv=10,scoring='accuracy')
logreg
logreg.mean()

from sklearn.model_selection import cross_val_score
svc=cross_val_score(SVC(),digit.data,digit.target, cv=10,scoring='accuracy')
svc
svc.mean()

from sklearn.model_selection import cross_val_score
rf=cross_val_score(RandomForestClassifier(n_estimators = 40),digit.data,digit.target, cv=10,scoring='accuracy')
rf
rf.mean()