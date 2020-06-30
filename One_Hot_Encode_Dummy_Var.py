# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:09:48 2019

@author: 140524
"""

#Dummy variable and One Hot encoding
#pandas - get_dummies
#sklearn - OneHotEncode

# use this to convert the name of the town to specific integers

#names of the towns are categorical - nominal variable which cannot be ordered
#if we assign 1 to Monroee, 2 to west windsowr, 3 to Robbinsville then it seems its seems 
#1>2>3 which should be the case

#thus single integer encoding will not work here
# we will use a onehot encoding to assign dummy variables to the town names

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/homeprices.csv')
df
#assign dummy variable by get dummies - this will return the dummy variable columns

dummies = pd.get_dummies(df.town)
dummies

#concating dummies with df

merged = pd.concat([df,dummies],axis=1)
merged

# dropping the original town column

final = merged.drop(['town','west windsor'],axis=1)
final
#we also want to drop one of the dummy columns because of dummy variable trap
#dummy variable trap - whenever one variable is derived from rest of the variables,
#the dummy variable columns are highly correlated, and becauae of being highly corelated it can mess up the
#machine learning model

#lenear regression model is aware of this trap but its a good practise to drop

#calling the linear reg model

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
#all the columns except price - see we have not set inplace =1
X = final.drop('price',axis=1)
y = final.price
reg.fit(X,y)

# predicting

reg.predict([[3300,1,0]])

#predicting for the dropped column west windsor

reg.predict([[2600,0,0]])

reg.score(X,y)
   
# using the oneHotencode
#in order to do onehot encode u have to use label encoding to your columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle
#using values so we get X as a 2D array instead of an array
X = df[['town','area']].values
X
y=df.price
y

#now creating dummy variables column

from sklearn.preprocessing import OneHotEncoder
#we have ti mention categorical feature so the hot encoder transform only 0th column
ohe = OneHotEncoder(categorical_features = [0])
#also converting it to an array
X = ohe.fit_transform(X).toarray()
X
X = X[:,1:]
X
reg.fit(X,y)
reg.predict([[1,0,2800]])


