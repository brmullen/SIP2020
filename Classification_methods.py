#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:40:33 2020

@author: saathvikdirisala
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb

'''data = pd.read_csv(r'data_banknote_authentication.csv')


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

plt.hist(X[0], alpha = 0.5)
plt.hist(X[1], alpha = 0.5)'''

#sb.distplot(y_test)
#sb.distplot(y_pred)

tsv_file = open('p21_classifier_ns_metrics.tsv')
read_tsv = pd.read_csv(tsv_file, delimiter="\t")
tsv_file.close()
data1 = read_tsv
#for row in read_tsv:
   # print(row)
#sb.pairplot(data1.iloc[:,1:5])
cols = set(data1.columns)
cols.remove('artifact')
cols.remove('signal')
cols.remove('exp_ic')
data2 = data1['signal']
data1 = data1[cols]

X = data1
y = data2

X = X.fillna(X.mean())

Split = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3)
for train_index, test_index in Split.split(X,y):
    X_test, X_train = X.iloc[test_index],X.iloc[train_index]
    y_test, y_train = y.iloc[test_index],y.iloc[train_index]
 
rnd_clf = RandomForestClassifier(n_estimators = 100, max_features = 4)
rnd_clf.fit(X_train,y_train)
y_pred = rnd_clf.predict(X_test)
print('recall:',recall_score(y_test,y_pred))
print('precision:',precision_score(y_test,y_pred))
print('default score:',rnd_clf.score(X_test,y_test))
conmat = confusion_matrix(y_test,y_pred)
print(conmat)

#sb.pairplot(X.iloc[3:5])
plt.hist(X.iloc[:,12], bins = 10, alpha = 0.5)
plt.hist(X.iloc[:,11], bins = 10, alpha = 0.5)

