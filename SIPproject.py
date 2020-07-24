#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:56:02 2020

@author: saathvikdirisala
"""


import pandas as pd
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from statistics import stdev
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.svm import SVR, SVC
import sys
import scipy as sp
import random

def read_tsv(file, index_col):
    tsv_file = open(file)
    data = pd.read_csv(tsv_file, delimiter = "\t", index_col = index_col)
    tsv_file.close()
    return data

for i in range(14):
    i = i+1
    if(i==1):
        data = read_tsv("/Users/saathvikdirisala/Desktop/python/PostNatalData/P"+str(i)+"_IC_metrics.tsv", index_col = "exp_ic")
    else:
        df = read_tsv("/Users/saathvikdirisala/Desktop/python/PostNatalData/P"+str(i)+"_IC_metrics.tsv", index_col = "exp_ic")
        data = data.append(df)

def missing_val_fill(data, col, filler):
    if len(str(filler))<=10:
        if filler == "random":
           for i in range(len(col)):
                data[col[i]][np.isnan(data[col[i]])==True] = list(np.random.normal(np.mean(data[col[i]][np.isnan(data[col[i]])==False]),
                                                                                   stdev(data[col[i]][np.isnan(data[col[i]])==False]),
                                                                                   len(data[col[i]][np.isnan(data[col[i]])==True])))
        else:
           data[col] = data[col].fillna(filler)
    else:
        data[col] = data[col].fillna(filler)
    return data

def FeatureScaler(df, num_cols, ScalerType):
    data = df[num_cols]
    if ScalerType == "Standard" or ScalerType == "standard" or ScalerType == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
    elif ScalerType == "MinMax" or ScalerType == "minmax" or ScalerType == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(data)
        data = scaler.transform(data)
    elif ScalerType == "Robust" or ScalerType == "robust" or ScalerType == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(data)
        data = scaler.transform(data)
    data = pd.DataFrame(data, index = df.index)
    data.columns = num_cols
    for i in num_cols:
        df[i] = data[i]
    return df

def Encoder(data, col):
    import collections
    a = collections.Counter(data[col])
    if(len(a)>2):
        lb = LabelBinarizer()
        cat = lb.fit_transform(data[col])
        cat = pd.DataFrame(cat, index = data.index)
        cat.columns = [list(lb.classes_)]
        data = data.drop(labels = [col], axis = 1)
        data[list(lb.classes_)] = cat
    else:
        lb = preprocessing.LabelEncoder()
        lb.fit(data[col])
        data[col] = lb.transform(data[col])
    return data

def cat_df(data, feature):
    unq = list(Counter(data[feature]))
    length = len(unq)
    data1 = list(np.zeros(length))
    for i in range(length):
        TF = data[feature]==unq[i]#.between(i*(length/split),(i+1)*(length/split))
        d = data[TF]
        col = list(d.columns)
        col.remove(feature) 
        data1[i] = d[col]
    return data1

def split(data, conditions):
    '''df = [0]
    for i in range(len(conditions)):
        df[i] = data[conditions[i]]
        df.append(0)
    df = pd.DataFrame(df[:-1])'''
    df = data[conditions]
    return df

def test_train_split(X,y, n_splits, test_size):
    Split = StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size)
    for train_index, test_index in Split.split(X,y):
        X_test, X_train = X.iloc[test_index],X.iloc[train_index]
        y_test, y_train = y.iloc[test_index],y.iloc[train_index]
    return X_train, X_test, y_train, y_test

num_cols = list(data.columns)
num_cols.remove("artifact")
num_cols.remove("signal")
num_cols.remove("anml")
num_cols.remove("age")

strat_samp = pd.DataFrame(columns = data.columns)
lnth = 0
for i in range(14):
    tmdf = pd.DataFrame(data[data["age"]==i+1])
    tmdf.index = list(sp.arange(0, len(data[data["age"]==i+1])))
    rand_index = random.sample(set(tmdf.index), 468)
    strat_samp = strat_samp.append(tmdf.iloc[rand_index])
    lnth = len(data[data["age"]==i+1])

#data = strat_samp

col = [0]
q = 0

col = list(data.columns)
col.remove("threshold.area")
col.remove("threshold.perc")

data[data["signal"]==1] = missing_val_fill(data[data["signal"]==1], col, data[data["signal"]==1].median())
data[data["signal"]==0] = missing_val_fill(data[data["signal"]==0], col, data[data["signal"]==0].median())


data = (data.pipe(missing_val_fill, col = ["threshold.area", "threshold.perc"], filler = 0)
            .pipe(split, conditions = data["age"].between(1,14)))
            #.pipe(split, conditions = data["threshold.area"]!=0)

data = [data]

for i in range(len(data)):
    data[i] = (data[i].pipe(FeatureScaler, num_cols = num_cols, ScalerType = "standard")
                      .pipe(Encoder, col = "artifact")
                      .pipe(Encoder, col = "signal"))
num_cols.append("age")

ages = 1
X = list(np.zeros(ages))
y = list(np.zeros(ages))
for i in range(ages):
    X[i] = data[i][num_cols]
    y[i] = data[i]["signal"]
    X[i].columns = num_cols
    y[i].columns = ["signal"]

X_train = list(np.zeros(ages))
X_test = list(np.zeros(ages))
y_train = list(np.zeros(ages))
y_test = list(np.zeros(ages))

# =============================================================================
# ranking_dfs = list(np.zeros(ages))
# fig = list(np.zeros(ages))
# for u in range(len(fig)):
#     fig[u] = list(np.zeros(len(X[0].columns)))
# for j in range(ages):
#     a = 0
#     shape_diff = list(np.zeros(len(X[j].columns)))
#     median_diff = list(np.zeros(len(X[j].columns)))
#     modal_diff = list(np.zeros(len(X[j].columns)))
#     skew_diff = list(np.zeros(len(X[j].columns)))
#     cluster_diff = list(np.zeros(len(X[j].columns)))
#     for i in X[j].columns:
#         signal_indices = y[j][y[j]==1].index
#         artifact_indices = y[j][y[j]==0].index
#         a+=1
#         sig = X[j][i][signal_indices]
#         art = X[j][i][artifact_indices]
#         median_diff[a-1] = abs(sig.median()-art.median())
#         if(stdev(sig)!=0 and stdev(art)!=0):
#             skew_diff[a-1] = abs(3*(sig.mean()-sig.median())/(stdev(sig)) - 3*(art.mean()-art.median())/(stdev(art)))
#         else:
#             skew_diff[a-1] = abs(3*(sig.mean()-sig.median()) - 3*(art.mean()-art.median()))
#   
#         plt.figure(100)
#         n = plt.hist([sig,art], bins = 20)
#         plt.close(fig=100)
#         plt.figure(a)
# #        plt.title(str(i) + "  " + str(median_diff[a-1]))
# #        sb.distplot(sig, bins = 20, hist = False, rug = True, color = "green", kde_kws = {'bw':0.1})
# #        sb.distplot(art, bins = 20, hist = False, rug = True, color = "red", kde_kws = {'bw':0.1})
#         fig[j][a-1] , ax1 = plt.subplots()
#         ax1.set_title("P"+str(j+1)+"  "+str(i))
#         ax1.hist(sig, bins = 50, alpha = 0.5, label ='artifact', color = 'r')
#         ax1.grid(False)         
#         ax1.set_xlabel('Values')
#         ax1.set_ylabel('# of instances', color = 'red')
#         
#         ax2 = ax1.twinx()
#         ax2.hist(art, bins = 50, alpha = 0.5, label ='signal', color = 'g')
#         ax2.set_ylabel('# of instances', color = 'green')
#         ax2.grid(False)
#         
#         ax1.legend(loc='upper left')
#         ax2.legend(loc='upper right')
#         #plt.show()
#         
#         temp_df = pd.DataFrame(n[0][0], columns = ["signal"])
#         temp_df["artifact"] = n[0][1]
#         loc = n[1]
#         temp_df = FeatureScaler(temp_df, ["signal", "artifact"], "minmax")
#         shape_diff[a-1] = list(np.array(temp_df["signal"])-np.array(temp_df["artifact"]))
#         modal_diff[a-1] = abs(loc[list(temp_df["signal"]).index(max(temp_df["signal"]))]-loc[list(temp_df["artifact"]).index(max(temp_df["artifact"]))])
#         cluster_diff[a-1] = modal_diff[a-1]*(1/stdev(sig))*(1/stdev(art))
#         
#     
#     def diff_gauge(data):
#         stdev = list(np.zeros(len(data)))
#         for i in range(len(data)):
#             stdev[i] = np.sqrt(sum(np.array(data[i])**2)/(len(data[i])-1))
#         return stdev
#     
#     diffg = diff_gauge(shape_diff)
#     
#     features = pd.DataFrame(X[j].columns, columns = ["feature"], index = np.arange(1,(len(X[j].columns)+1)))
#     features["median_diff"] = median_diff
#     features["shape_diff"] = diffg
#     features["modal_diff"] = modal_diff
#     features["skew_diff"] = skew_diff
#     features["cluster_diff"] = cluster_diff
#     features = FeatureScaler(features, ["median_diff", "shape_diff", "modal_diff", "skew_diff", "cluster_diff"], "minmax")
#     #features["combined_gauge1"] = np.array(features["median_diff"])+np.array(features["shape_diff"])+np.array(features["modal_diff"])
#     #features["combined_gauge2"] = (np.array(features["spread_diff"]))+np.array(features["median_diff"])
#     
#     ranking_dfs[j] = features
#     
# r = len(ranking_dfs[0])
# rank = list(np.zeros(len(ranking_dfs)))
# for h in range(len(ranking_dfs)):
#     ranking_dfs[h]["Combo_Gauge"] = 3*np.array(ranking_dfs[h]["cluster_diff"]) + 1*np.array(ranking_dfs[h]["median_diff"]) + 0*np.array(ranking_dfs[h]["shape_diff"]) + 2.3*np.array(ranking_dfs[h]["modal_diff"]) + 0*np.array(ranking_dfs[h]["skew_diff"])
#     ranking_dfs[h] = (ranking_dfs[h].sort_values(by = ["cluster_diff"], ascending = False ,ignore_index = True))
#     rank[h] = ranking_dfs[h][:r]
# 
# print(rank)
# 
# def figure(data, feature):
#     index = list(data.columns).index(feature)
#     return index
# 
# a = [0]
# b = 0
# for i in range(len(rank)):
#     for j in (rank[i]["feature"]):
#         idx = figure(X[0], j) 
#         a[b] = fig[i][idx]
#         b+=1
#         a.append(0)
# a = a[:-1]
# =============================================================================

sampsize = 100
lst = list(np.zeros(ages))
for t in range(ages):
    lst[t] = list(np.zeros(sampsize))

scores = pd.DataFrame(list(range(1,(ages+1))), columns = ["NoDomain"])
scores["Accuracy"] = lst
scores["Precision"] = lst
scores["Recall"] = lst
_params = pd.DataFrame(list(range(1,(ages+1))), columns = ["Age"])
_params["params"] = lst


imp_features = ['temporal.autocorr', "freq.rangesz",'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max', 'age']
'''['temporal.autocorr',
 'temporal.min',
 'region.extent',
 'region.majaxis',
 'region.majmin.ratio',
 "region.minaxis",
 "mass.region",
 "threshold.area",
 "mass.total",
 "freq.rangesz", 
 "freq.maxsnr.freq", 
 "freq.avgsnr", "temporal.max", "age"]'''
#['temporal.autocorr', 'region.extent', 'mass.total', 'freq.rangesz', 'freq.avgsnr', 'age', 'mass.region']#['temporal.autocorr', 'region.extent', 'region.minaxis', 'threshold.area', 'mass.total', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'age']#['temporal.autocorr', 'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max', 'age']

#["region.minaxis", "region.majaxis", "mass.total", "threshold.area", "mass.region", "spatial.min", "freq.rangesz", "freq.maxsnr.freq", "freq.avgsnr", "temporal.max", "age"] #['region.majaxis', 'threshold.area', 'mass.region', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max', 'age']
# "temporal.max", "spatial.min", 'freq.range.high',  'freq.range.low',
for i in range(ages):
            scores = pd.DataFrame(list(np.zeros(sampsize)), columns = ["Accuracy"])
            scores["Precision"] = list(np.zeros(sampsize))
            scores["Recall"] = list(np.zeros(sampsize))
            for h in range(sampsize):
                    X_train, X_test, y_train, y_test = test_train_split(X[i],y[i], 10, 0.3)
                    X_train_mod = X_train[imp_features]
                    X_test_mod = X_test[imp_features]
                    
                    '''ann = tf.keras.models.Sequential()
                    ann.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))
                    ann.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))
                    ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
                    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                    ann.fit(X_train_mod, y_train, batch_size = 20, epochs = 3)
                    y_pred = ann.predict(X_test_mod)
                    y_pred = (y_pred>0.5)'''
                    
                    '''log = LogisticRegression(solver = "lbfgs")
                    log.fit(X_train_mod, y_train)
                    y_pred = log.predict(X_test_mod)'''
                    
                    '''gnb = GaussianNB(var_smoothing = 0.0001)
                    gnb.fit(X_train_mod, y_train)
                    y_pred = gnb.predict(X_test_mod)'''
                    
                    '''svc = SVC(kernel = "linear", probability = True)
                    svc.fit(X_train_mod, y_train)
                    y_pred = svc.predict(X_test_mod)'''
                    
                    rnf = RandomForestClassifier(n_estimators = 18, max_features = 5, class_weight = {0:1,1:1})
                    rnf.fit(X_train_mod, y_train)
                    y_pred = rnf.predict(X_test_mod)
                    
                    '''rnf1 = RandomForestClassifier(n_estimators = 10, max_features = 3, class_weight = {0:3.711,1:1})
                    
                    rnf2 = RandomForestClassifier(n_estimators = 20, max_features = 5, class_weight = {0:3.711,1:1})
                    
                    rnf3 = RandomForestClassifier(n_estimators = 20, max_features = 3, class_weight = {0:3.5,1:1})
                    '''
                    
                    '''voter = VotingClassifier(estimators = [("rnf", rnf), ("rnf1", rnf1), ("rnf2", rnf2), ("rnf3", rnf3)], voting = "soft")
                    voter.fit(X_train_mod, y_train)
                    y_pred = voter.predict(X_test_mod)'''
                    
                    '''n_estimators = [16,17,18,19]
                    max_features = [4,5,7,8]
                    class_weight = [{0:3,1:1}, {0:4,1:1}, {0:3.5,1:1}, {0:3.711,1:1}, {0:1,1:1}, {0:2,1:1}]
        
                    param_grid = dict(n_estimators = n_estimators,
                                      max_features = max_features,
                                      class_weight = class_weight)
                                      #criterion = criterion)
                    
                    grid = GridSearchCV(estimator = rnf,
                                        param_grid = param_grid,
                                        scoring = "f1",
                                        verbose = 1,
                                        n_jobs = -1)
                    
                    grid_result = grid.fit(X_train_mod, y_train)
                    
                    print("Best Score:", grid_result.best_score_)
                    print("Best Parameters:", grid_result.best_params_)'''
                    
                    scores["Accuracy"][h] = accuracy_score(y_test, y_pred)
                    scores["Precision"][h] = precision_score(y_test, y_pred)
                    scores["Recall"][h] = recall_score(y_test, y_pred)
                    print(i,h, sep = " ")
                
            scrdata = {"Accuracy":scores["Accuracy"], "Precision":scores["Precision"], "Recall":scores["Recall"]}
            fig1, ax1 = plt.subplots()
            ax1.boxplot(scrdata.values())
            ax1.set_xticklabels(scrdata.keys())
            plt.title(str(i)+" "+str(q))
            plt.ylim(0.90,1.0)
            
            
# =============================================================================
# sampsize = 25
# lst = list(np.zeros(ages))
# for t in range(ages):
#     lst[t] = list(np.zeros(sampsize))
# 
# scores = pd.DataFrame(list(range(1,(ages+1))), columns = ["NoDomain"])
# scores["Accuracy"] = lst
# scores["Precision"] = lst
# scores["Recall"] = lst
# _params = pd.DataFrame(list(range(1,(ages+1))), columns = ["Age"])
# _params["params"] = lst
# 
# 
# imp_features = ['temporal.autocorr', 'region.extent', 'region.minaxis', 'threshold.area', 'mass.total', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'age', "region.majaxis", "mass.region", "temporal.max"]
# '''['temporal.autocorr',
#  'temporal.min',
#  'region.extent',
#  'region.majaxis',
#  'region.majmin.ratio',
#  "region.minaxis",
#  "mass.region",
#  "threshold.area",
#  "mass.total",
#  "freq.rangesz", 
#  "freq.maxsnr.freq", 
#  "freq.avgsnr", "temporal.max", "age"]'''
# #['freq.range.low', 'region.majaxis', 'region.majmin.ratio', 'temporal.autocorr', 'temporal.min', 'region.minaxis', 'mass.total', 'spatial.min', 'freq.avgsnr', 'temporal.max']
# 
#  #["region.minaxis", "region.majaxis", "mass.total", "threshold.area", "mass.region", "spatial.min", "freq.rangesz", "freq.maxsnr.freq", "freq.avgsnr", "temporal.max", "age"]
# 
# for i in range(ages):
#         new_features = list(imp_features)
#         for e in range(len(new_features)):
#             scores = pd.DataFrame(list(np.zeros(sampsize)), columns = ["Accuracy"])
#             scores["Precision"] = list(np.zeros(sampsize))
#             scores["Recall"] = list(np.zeros(sampsize))
#             performance = []
#             print(new_features)
#             print(len(new_features))
#             for q in range(len(new_features)+1):
#                 features = list(new_features)
#                 if q<len(new_features):
#                     features.remove(features[q])
#                 for h in range(sampsize):
#                     X_train, X_test, y_train, y_test = test_train_split(X[i],y[i], 10, 0.3)
#                     X_train_mod = X_train[imp_features]
#                     X_test_mod = X_test[imp_features]
#                     
#                     '''ann = tf.keras.models.Sequential()
#                     ann.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))
#                     ann.add(tf.keras.layers.Dense(units = 9, activation = 'relu'))
#                     ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
#                     ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#                     ann.fit(X_train_mod, y_train, batch_size = 20, epochs = 3)
#                     y_pred = ann.predict(X_test_mod)
#                     y_pred = (y_pred>0.5)'''
#                     
#                     '''log = LogisticRegression(solver = "lbfgs")
#                     log.fit(X_train_mod, y_train)
#                     y_pred = log.predict(X_test_mod)'''
#                     
#                     '''gnb = GaussianNB(var_smoothing = 0.0001)
#                     gnb.fit(X_train_mod, y_train)
#                     y_pred = gnb.predict(X_test_mod)'''
#                     
#                     '''svc = SVC(kernel = "linear", probability = True)
#                     svc.fit(X_train_mod, y_train)
#                     y_pred = svc.predict(X_test_mod)'''
#                     
#                     if len(features)>=5:
#                         rnf = RandomForestClassifier(n_estimators = 18, max_features = 5, class_weight = {0:1,1:1})
#                     else:
#                         rnf = RandomForestClassifier(n_estimators = 18, max_features = len(features), class_weight = {0:1,1:1})
#                     rnf.fit(X_train_mod, y_train)
#                     y_pred = rnf.predict(X_test_mod)
#                      
#                     
#                     '''rnf1 = RandomForestClassifier(n_estimators = 10, max_features = 3, class_weight = {0:3.711,1:1})
#                     
#                     rnf2 = RandomForestClassifier(n_estimators = 20, max_features = 5, class_weight = {0:3.711,1:1})
#                     
#                     rnf3 = RandomForestClassifier(n_estimators = 20, max_features = 3, class_weight = {0:3.5,1:1})
#                     '''
#                     
#                     '''voter = VotingClassifier(estimators = [("rnf", rnf), ("rnf1", rnf1), ("rnf2", rnf2), ("rnf3", rnf3)], voting = "soft")
#                     voter.fit(X_train_mod, y_train)
#                     y_pred = voter.predict(X_test_mod)'''
#                     
#                     '''n_estimators = [16,17,18,19]
#                     max_features = [4,5,7,8]
#                     class_weight = [{0:3,1:1}, {0:4,1:1}, {0:3.5,1:1}, {0:3.711,1:1}, {0:1,1:1}, {0:2,1:1}]
#                     #criterion = ["gini","entropy"]
#         
#                     param_grid = dict(n_estimators = n_estimators,
#                                       max_features = max_features,
#                                       class_weight = class_weight)
#                                       #criterion = criterion)
#                     
#                     grid = GridSearchCV(estimator = rnf,
#                                         param_grid = param_grid,
#                                         scoring = "f1",
#                                         verbose = 1,
#                                         n_jobs = -1)
#                     
#                     grid_result = grid.fit(X_train_mod, y_train)
#                     
#                     print("Best Score:", grid_result.best_score_)
#                     print("Best Parameters:", grid_result.best_params_)
#                     '''
# 
#                     scores["Accuracy"][h] = accuracy_score(y_test, y_pred)
#                     scores["Precision"][h] = precision_score(y_test, y_pred)
#                     scores["Recall"][h] = recall_score(y_test, y_pred)
#                     print(i,e,q,h, sep = " ")
#                 
#                 scrdata = {"Accuracy":scores["Accuracy"], "Precision":scores["Precision"], "Recall":scores["Recall"]}
#                 performance.append(np.mean(scores)["Accuracy"])
#                 
#                 print(np.mean(scores)["Accuracy"])
#                 
#                 '''fig1, ax1 = plt.subplots()
#                 ax1.boxplot(scrdata.values())
#                 ax1.set_xticklabels(scrdata.keys())
#                 plt.title(str(i)+" "+str(q))'''
#             performance = [round(num, ndigits = 5) for num in performance]
#             print(performance)
#             if performance.index(max(performance))<len(new_features):
#                 new_features.remove(new_features[performance.index(max(performance))])
#             else:
#                 new_features = new_features
#                 print("No changes ahead")
#                 print(new_features)
#                 sys.exit()
# =============================================================================
            
        

    
# =============================================================================
# Only Domain set/RNF-17-7-3.711:1:
# new_features = ['region.majaxis', 'threshold.area', 'mass.region', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max', 'age']
#                ['region.majaxis', 'threshold.area', 'mass.region', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max', 'age']
# Only Domain set/RNF-18-5-1:1:              
# new_features = ['region.minaxis', 'region.majaxis', 'mass.total', 'mass.region', 'age']
# =============================================================================
# Full set:
#new_features = ['region.minaxis', 'region.majaxis', 'mass.total', 'threshold.area', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'age']
# =============================================================================
# Only age 1:
#['freq.avgsnr',
#  'freq.maxsnr',
#  'freq.range.high',
#  'freq.range.low',
#  'freq.rangesz',
#  'length',
#  'mass.perc',
#  'mass.region',
#  'region.centroid.0',
#  'region.extent',
#  'region.majaxis',
#  'region.majmin.ratio',
#  'region.minaxis',
#  'spatial.COMall.y',
#  'spatial.avg',
#  'spatial.max',
#  'spatial.min',
#  'spatial.std',
#  'temporal.autocorr',
#  'temporal.min',
#  'temporal.n.freq',
#  'temporal.std',
#  'threshold.area']
# =============================================================================
# =============================================================================
# pca = PCA(n_components = 2)
# pca.fit(X[0])
# df3 = pd.DataFrame(pca.components_)
# dfmat = np.matrix(df3).T
# datamat = np.matrix(X[0])
# pca_trans = pd.DataFrame(datamat @ dfmat)
# 
# plt.scatter(pca_trans.iloc[:,0], pca_trans.iloc[:,1], c = y[0], alpha = 0.5)
# 
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(pca_trans.iloc[:,0], pca_trans.iloc[:,1], pca_trans.iloc[:,2], c = y[0], marker = "^", alpha = 0.3)
# ax.view_init(elev = 0, azim = 180)
# plt.show()
# =============================================================================



# =============================================================================
# for n_comp in sp.arange(1,4):
#     
#     svd = sp.linalg.svd(X[0], full_matrices = False)
#     sigma = np.zeros((n_comp, n_comp))
#     for i in range(n_comp):
#         sigma[i,i] = svd[1][i]
#     a1 = svd[0][:, 0:n_comp] @ sigma @ svd[2][0:n_comp, :]
#     clms = []
#     for j in range(len(X[0].columns)):
#         clms.append("SVD"+str(j+1))
#     adf = pd.DataFrame(a1)
#     adf.columns = clms
#     adf.index = X[0].index
#     '''plt.figure(1)
#     plt.scatter(adf["SVD1"], adf["SVD2"], c = y[0], cmap = "Spectral", alpha = 0.3)
#     plt.xlabel("SVD1")
#     plt.ylabel("SVD2")
#     plt.legend()'''
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection = "3d")
#     ax.scatter(adf["SVD1"], adf["SVD2"], adf["SVD3"], c = y[0], marker = "^", alpha = 0.3)
#     ax.view_init(elev = 50, azim = 130)
#     plt.show()
# =============================================================================

# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(adf.iloc[:,0], adf.iloc[:,1], adf.iloc[:,2], c = y[0], marker = "^", alpha = 0.3)
# ax.view_init(elev = 45, azim = 45)
# plt.show()
# =============================================================================

# =============================================================================
# 
# ColorMap shades:
#[Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r]
# 
# =============================================================================

'''
wave1 = []
wave2 = []
for i in range(50):
    wave1.append(5*np.sin(i/5)-15)
    wave2.append(np.tan(i/5)/1.5 + 15)
plt.figure(1)
plt.plot(sp.arange(0,10,1/5), wave1)
plt.plot(sp.arange(0,10,1/5), wave2)
plt.xlim(0,10)
plt.ylim(-25,28)

wave12 = np.array(wave1) + np.array(wave2)
wave21 = 0.254*np.array(wave1) + 1.12*np.array(wave2)

plt.figure(2)
plt.plot(sp.arange(0,10,1/5), wave21, c = "maroon")
plt.xlim(0,10)
plt.ylim(-15,30)

plt.scatter(X[0]["spatial.COMdom.x"], X[0]["spatial.COMdom.y"],)'''

'''
from matplotlib.image import img
Xt = X[0]["spatial.COMdom.y"]
yt = X[0]["spatial.COMdom.x"]
Xt = 104*Xt+320
yt = 140*yt+400

rd = img.imread("/Users/saathvikdirisala/Downloads/brain_outline.jpg")
image = np.mean(rd,-1)
plt.figure(1)
plt.imshow(image, cmap = "rainbow")
plt.scatter(Xt, yt, alpha = 0.3)
plt.xlim(0,600)
plt.ylim(0,800)'''


#['temporal.autocorr', 'region.extent', 'region.majaxis', 'region.minaxis', 'mass.region', 'threshold.area', 'freq.maxsnr.freq', 'freq.avgsnr', 'temporal.max', 'age']
#['temporal.autocorr', 'region.extent', 'region.minaxis', 'threshold.area', 'mass.total', 'freq.rangesz', 'freq.maxsnr.freq', 'freq.avgsnr', 'age']
#['temporal.autocorr', 'region.extent', 'mass.total', 'freq.rangesz', 'freq.avgsnr', 'age', 'mass.region']


