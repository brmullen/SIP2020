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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random

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

'''bestfeatures = SelectKBest(score_func = chi2, k=36)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(20, 'Score'))
'''

scr = [0]*len(X.columns)
for j in range(10):
    Split = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3)
    for train_index, test_index in Split.split(X,y):
        X_test, X_train = X.iloc[test_index],X.iloc[train_index]
        y_test, y_train = y.iloc[test_index],y.iloc[train_index]
    
    #for j in range(36):
  
    t = 0
    for i in X.columns:
            cols = set(X.columns)
            cols.remove(i)
            trnX = X_train[cols]
            tstX = X_test[cols]
            rnd_clf = RandomForestClassifier(n_estimators = 100, max_features = 4)
            rnd_clf.fit(trnX,y_train)
            scr[t] = rnd_clf.score(tstX,y_test) + scr[t]
            t+=1
            print(j,i)
    print(scr)
    data = np.concatenate((np.array(X.columns).reshape(len(X.columns),1),np.array(scr).reshape(len(scr),1)),1)
    data = pd.DataFrame(data)
    data.columns = ["Feature", "Performance w/o"]


data.sort_values(by=["Performance w/o"], inplace = True, ignore_index = True)



    #cl = set(X.columns)
    #cl.remove(X.columns[scr.index(max(scr))])
    #X = X[cl]
    #X_train = X_train[cl]
    #X_test = X_test[cl]
    #ordr[j] = X.columns[scr.index(max(scr))]
    #ordr.append(1)


'''y_pred = rnd_clf.predict(X_test)
print('recall:',recall_score(y_test,y_pred))
print('precision:',precision_score(y_test,y_pred))
print('default score:',rnd_clf.score(X_test,y_test))
conmat = confusion_matrix(y_test,y_pred)
print(conmat)

#sb.pairplot(X.iloc[3:5])
plt.hist(X.iloc[:,12], bins = 10, alpha = 0.5)
plt.hist(X.iloc[:,11], bins = 10, alpha = 0.5)'''


'''
                Feature Performance w/o
0           spatial.max        0.940722
1      spatial.COMall.y        0.949589
2          temporal.std        0.949754
3           mass.region        0.950082
4             mass.perc        0.950411
5        region.minaxis        0.950575
6        region.majaxis        0.950739
7           spatial.avg        0.950739
8          temporal.max        0.950903
9           spatial.min        0.950903
10         freq.rangesz        0.951067
11       threshold.area        0.951067
12          freq.maxsnr        0.951067
13      freq.range.high        0.951232
14  region.majmin.ratio        0.951396
15    region.centroid.1        0.951396
16         temporal.min         0.95156
17           mass.total         0.95156
18                  age         0.95156
19     spatial.COMdom.y         0.95156
20    spatial.n.domains        0.951724
21     spatial.COMdom.x        0.951724
22    region.centroid.0        0.951888
23       threshold.perc        0.951888
24               length        0.951888
25     freq.maxsnr.freq        0.952217
26       freq.integrate        0.952217
27     spatial.COMall.x        0.952217
28      temporal.n.freq        0.952381
29          spatial.std        0.952545
30        region.extent        0.952545
31        region.orient        0.953366
32       freq.range.low        0.953695
33    temporal.autocorr        0.953695
34          freq.avgsnr        0.953859
35  region.eccentricity        0.954351
'''