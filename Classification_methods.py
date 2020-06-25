#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:40:33 2020

@author: saathvikdirisala
"""


import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score
from sklearn import preprocessing

data = pd.read_csv(r'data_banknote_authentication.csv')


X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

Split = StratifiedShuffleSplit(n_splits = 10, test_size = 0.25)
for train_index, test_index in Split.split(X,y):
    X_test, X_train = X.iloc[test_index], X.iloc[train_index]
    y_test, y_train = y.iloc[test_index], y.iloc[train_index]
    
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

gnb_clf = GaussianNB(var_smoothing = 0.00001)
gnb_clf.fit(X_train,y_train)
y_pred = gnb_clf.predict(X_test)
print('recall:',recall_score(y_pred,y_test))
print('precision:',precision_score(y_pred,y_test))
print('default:',gnb_clf.score(X_test,y_test))
conmat = confusion_matrix(y_test,y_pred)
print(conmat)

rnd_clf = RandomForestClassifier(n_estimators = 50, max_features = 3)
rnd_clf.fit(X_train,y_train)
y_pred1 = rnd_clf.predict(X_test)
print('recall:',recall_score(y_pred1,y_test))
print('precision;',precision_score(y_pred1,y_test))
print('default:',rnd_clf.score(X_test,y_test))
conmat1 = confusion_matrix(y_test,y_pred1)
print(conmat1)

