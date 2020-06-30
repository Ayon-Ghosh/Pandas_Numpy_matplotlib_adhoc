# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:52:39 2019

@author: 140524
"""

#Feature Selection
#https://www.youtube.com/watch?v=EqLBAmtKMnQ&list=PLZoTAELRMXVOnN_g96ayzXX5i7RRO0QhL&index=44

#Univariate Selection

# =============================================================================
# Statistical tests can be used to select those features that have the strongest 
# relationship with the output variable.
# 
# The scikit-learn library provides the SelectKBest class that can be used with a 
# suite of different statistical tests to select a specific number of features.
# 
# The example below uses the chi-squared (chi²) statistical test for non-negative 
# features to select 10 of the best features from the Mobile Price Range Prediction Dataset.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
train = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Feature-Selection-techniques/master/train.csv')
train.head()
X=train.iloc[:,0:20]
X.columns
X.columns.values
y=train[['price_range']]
y
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame({'SKB_scores':fit.scores_})
dfscores
dfcolumns = pd.DataFrame({'features':X.columns})
dfcolumns

featurescores = pd.concat([dfcolumns,dfscores],axis=1)
featurescores.head()
featurescore_sorted = featurescores.sort_values('SKB_scores', ascending = False)
featurescore_sorted.nlargest(10,'SKB_scores')
SKB = featurescores.set_index('features')
SKB


# =============================================================================
# Feature Importance
# You can get the feature importance of each feature of your dataset by using the 
# feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher 
# the score more important or relevant is the feature towards your output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Classifiers, 
# we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.
# =============================================================================

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)
feature_imp = pd.DataFrame({'ETC_score':model.feature_importances_})
feature_imp
dfcolumns = pd.DataFrame({'features':X.columns})
dfcolumns
featureimp = pd.concat([dfcolumns,feature_imp],axis=1)
featureimp.set_index('features',inplace=True)
featureimp
featureimp.plot(kind='barh')
plt.show()
ETC = featureimp
ETC
# =============================================================================
# Correlation Matrix with Heatmap
# Correlation states how the features are related to each other or the target variable.
# 
# Correlation can be positive (increase in one value of feature increases the value 
# of the target variable) or negative (increase in one value of feature decreases the 
# value of the target variable)
# 
# Heatmap makes it easy to identify which features are most related to the target 
# variable, we will plot heatmap of correlated features using the seaborn library.
# =============================================================================


import seaborn as sns
#get correlations of each features in dataset
corrmat = train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Feature selection using RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,y)
print(clf.feature_importances_)
featureimp = pd.DataFrame({'RFC_score':clf.feature_importances_})
featureimp
dfcolumns = pd.DataFrame({'features':X.columns})
dfcolumns
featureimp = pd.concat([dfcolumns,featureimp],axis=1)
featureimp.set_index('features',inplace=True)
featureimp
featureimp.plot(kind='barh')
plt.show()
RFC = featureimp
RFC

#Recursive Feature Elemination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

mod = LogisticRegression()
rfe = RFE(mod, 20)
rfe_fit = rfe.fit(X, y)
rfe.support_
featureRFE = pd.DataFrame({'RFE_score':rfe.support_})
featureRFE
RFE = pd.concat([dfcolumns,featureRFE],axis=1)
RFE = RFE.set_index('features')
RFE 
# Combine all feature section together

from functools import reduce
fs = [RFC,RFE,SKB,ETC]
final_results = reduce(lambda x,y: pd.merge(x,y,on ='features'),fs)
final_results

final_results['RFE_score'] = final_results['RFE_score'].astype(int)
final_results

final_results['total_score'] = final_results['RFC_score']+final_results['RFE_score']+final_results['SKB_scores']+final_results['ETC_score']
final_results=final_results.sort_values('total_score',ascending = False)
final_results
